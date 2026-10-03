"""
TikTok Profile & Video Data Scraper
====================================
WORKING methods to get full TikTok profile data and video content.
Tested and verified April 2026.

METHODS SUMMARY:
================
METHOD 1 (BEST): HTML SSR Scraping - Parse __UNIVERSAL_DATA_FOR_REHYDRATION__
  - Gets: FULL profile (bio, stats, follower/following/heart/video counts)
  - Works: YES - Reliable, no auth needed, just curl + parse
  - Limitation: No video list on profile page (videos load client-side)

METHOD 2: oEmbed API - https://www.tiktok.com/oembed?url=...
  - Gets: Video title/caption, author, thumbnail URL
  - Works: YES - No auth, no rate limit issues
  - Limitation: Need video IDs first; no engagement stats

METHOD 3: tikwm.com API - https://www.tikwm.com/api/
  - Gets: Full user info + individual video stats (plays, likes, comments, shares)
  - User info: https://www.tikwm.com/api/user/info?unique_id={username}
  - Video info: https://www.tikwm.com/api/?url={tiktok_video_url}
  - Works: YES for user info and single videos
  - Limitation: Posts list endpoint returns 403 (rate-limited)

METHOD 4: Video ID Discovery via Search Engines
  - Use web_search("site:tiktok.com/@{username}/video") to find video IDs
  - Then use oEmbed or tikwm or HTML scraping per video
  - Works: YES - Gets ~5 recent video IDs per search

METHOD 5: SocialBlade via web_extract
  - URL: https://socialblade.com/tiktok/user/{username}
  - Gets: Followers, following, likes, videos, growth trends, rankings
  - Works: YES via web_extract tool

METHOD 6: Individual Video HTML Scraping
  - Fetch https://www.tiktok.com/@{user}/video/{id}
  - Parse __UNIVERSAL_DATA webapp.video-detail -> itemInfo.itemStruct
  - Gets: FULL video data (caption, stats, music, hashtags, duration)
  - Works: YES - Most complete per-video data

NOT WORKING:
  - TikTok /api/user/detail/ endpoint -> returns empty (needs signed params)
  - TikTok /api/post/item_list/ -> returns empty (needs x-bogus/msToken)
  - tikwm.com /api/user/posts -> 403 forbidden
"""

import re
import json
import subprocess
import urllib.parse

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'


def fetch_url(url, headers=None):
    """Fetch URL via curl and return content."""
    cmd = ['curl', '-s', '-L', '-m', '30', url,
           '-H', f'User-Agent: {USER_AGENT}',
           '-H', 'Accept-Language: en-US,en;q=0.9']
    if headers:
        for k, v in headers.items():
            cmd.extend(['-H', f'{k}: {v}'])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    return result.stdout


def method1_html_profile(username):
    """
    METHOD 1: Scrape TikTok profile HTML and parse SSR JSON data.
    Returns full profile with stats.
    """
    url = f'https://www.tiktok.com/@{username}'
    html = fetch_url(url)

    m = re.search(
        r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">(.*?)</script>',
        html
    )
    if not m:
        return None

    data = json.loads(m.group(1))
    scope = data.get('__DEFAULT_SCOPE__', {})
    user_detail = scope.get('webapp.user-detail', {})
    user_info = user_detail.get('userInfo', {})

    if not user_info:
        return None

    user = user_info.get('user', {})
    stats = user_info.get('statsV2', user_info.get('stats', {}))

    return {
        'id': user.get('id'),
        'username': user.get('uniqueId'),
        'nickname': user.get('nickname'),
        'bio': user.get('signature'),
        'verified': user.get('verified'),
        'private': user.get('privateAccount'),
        'secUid': user.get('secUid'),
        'avatarLarger': user.get('avatarLarger'),
        'bioLink': user.get('bioLink', {}),
        'createTime': user.get('createTime'),
        'language': user.get('language'),
        'stats': {
            'followers': int(stats.get('followerCount', 0)),
            'following': int(stats.get('followingCount', 0)),
            'hearts': int(stats.get('heartCount', 0)),
            'videos': int(stats.get('videoCount', 0)),
            'diggs': int(stats.get('diggCount', 0)),
            'friends': int(stats.get('friendCount', 0)),
        }
    }


def method2_oembed_video(username, video_id):
    """
    METHOD 2: Get video caption/title via oEmbed.
    No auth needed. Returns caption, author, thumbnail.
    """
    url = f'https://www.tiktok.com/oembed?url=https://www.tiktok.com/@{username}/video/{video_id}'
    content = fetch_url(url)
    try:
        data = json.loads(content)
        return {
            'video_id': video_id,
            'title': data.get('title', ''),
            'author_name': data.get('author_name'),
            'author_url': data.get('author_url'),
            'thumbnail_url': data.get('thumbnail_url'),
            'thumbnail_width': data.get('thumbnail_width'),
            'thumbnail_height': data.get('thumbnail_height'),
        }
    except json.JSONDecodeError:
        return None


def method3_tikwm_user(username):
    """
    METHOD 3a: Get user info via tikwm.com API.
    """
    url = f'https://www.tikwm.com/api/user/info?unique_id={username}'
    content = fetch_url(url)
    try:
        data = json.loads(content)
        if data.get('code') == 0:
            return data['data']
    except json.JSONDecodeError:
        pass
    return None


def method3_tikwm_video(video_url):
    """
    METHOD 3b: Get video details via tikwm.com API.
    Returns: title, play_count, digg_count, comment_count, share_count, duration, download URLs
    """
    url = f'https://www.tikwm.com/api/?url={urllib.parse.quote(video_url)}'
    content = fetch_url(url)
    try:
        data = json.loads(content)
        if data.get('code') == 0:
            v = data['data']
            return {
                'video_id': v.get('id'),
                'title': v.get('title'),
                'duration': v.get('duration'),
                'play_count': v.get('play_count'),
                'likes': v.get('digg_count'),
                'comments': v.get('comment_count'),
                'shares': v.get('share_count'),
                'author': v.get('author', {}).get('unique_id'),
                'music_title': v.get('music_info', {}).get('title') if v.get('music_info') else None,
                'cover_url': v.get('origin_cover') or v.get('cover'),
                'play_url': v.get('play'),  # direct video URL
            }
    except json.JSONDecodeError:
        pass
    return None


def method6_html_video(username, video_id):
    """
    METHOD 6: Scrape individual video page HTML for full data.
    Gets: caption, full stats, music, hashtags, create time.
    """
    url = f'https://www.tiktok.com/@{username}/video/{video_id}'
    html = fetch_url(url)

    m = re.search(
        r'<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">(.*?)</script>',
        html
    )
    if not m:
        return None

    data = json.loads(m.group(1))
    scope = data.get('__DEFAULT_SCOPE__', {})
    vd = scope.get('webapp.video-detail', {})
    item = vd.get('itemInfo', {}).get('itemStruct', {})

    if not item:
        return None

    stats = item.get('statsV2', item.get('stats', {}))
    music = item.get('music', {})
    challenges = item.get('challenges', [])

    return {
        'video_id': item.get('id'),
        'description': item.get('desc'),
        'createTime': item.get('createTime'),
        'duration': item.get('video', {}).get('duration'),
        'stats': {
            'plays': int(stats.get('playCount', 0)),
            'likes': int(stats.get('diggCount', 0)),
            'comments': int(stats.get('commentCount', 0)),
            'shares': int(stats.get('shareCount', 0)),
            'saves': int(stats.get('collectCount', 0)),
        },
        'music': {
            'title': music.get('title'),
            'author': music.get('authorName'),
        },
        'hashtags': [c.get('title', '') for c in challenges],
        'author': item.get('author', {}).get('uniqueId'),
    }


def get_full_tiktok_profile(username):
    """
    Complete pipeline: Get full profile + discover and scrape recent videos.
    
    Returns dict with profile data, stats, and recent video details.
    """
    # Step 1: Get profile data
    profile = method1_html_profile(username)
    if not profile:
        return {'error': f'Could not fetch profile for @{username}'}

    result = {
        'profile': profile,
        'videos': [],
        'data_sources': ['tiktok_html_ssr'],
    }

    # Note: Video discovery requires web_search tool (not available in pure Python)
    # In the agent context, use:
    #   web_search(f"site:tiktok.com/@{username}/video")
    # Then for each video ID found, call method6_html_video() or method2_oembed_video()
    
    return result


if __name__ == '__main__':
    import sys
    username = sys.argv[1] if len(sys.argv) > 1 else 'khaby.lame'
    
    print(f'=== Testing TikTok scraping for @{username} ===\n')
    
    print('--- METHOD 1: HTML Profile Scraping ---')
    profile = method1_html_profile(username)
    if profile:
        print(f'  Username: {profile["username"]}')
        print(f'  Nickname: {profile["nickname"]}')
        print(f'  Bio: {profile["bio"][:100]}')
        print(f'  Verified: {profile["verified"]}')
        print(f'  Followers: {profile["stats"]["followers"]:,}')
        print(f'  Following: {profile["stats"]["following"]:,}')
        print(f'  Hearts: {profile["stats"]["hearts"]:,}')
        print(f'  Videos: {profile["stats"]["videos"]:,}')
        print(f'  SecUid: {profile["secUid"][:50]}...')
    else:
        print('  FAILED')
    
    print('\n--- METHOD 3a: tikwm.com User API ---')
    tikwm_user = method3_tikwm_user(username)
    if tikwm_user:
        s = tikwm_user.get('stats', {})
        print(f'  Followers: {s.get("followerCount"):,}')
        print(f'  Hearts: {s.get("heartCount"):,}')
        print(f'  Videos: {s.get("videoCount"):,}')
    else:
        print('  FAILED')
    
    # Test with a known video
    test_video_id = '7615318641042623775'  # khaby birthday video
    if username == 'khaby.lame':
        print(f'\n--- METHOD 2: oEmbed for video {test_video_id} ---')
        oembed = method2_oembed_video(username, test_video_id)
        if oembed:
            print(f'  Title: {oembed["title"][:80]}')
        
        print(f'\n--- METHOD 6: HTML Video Scraping for {test_video_id} ---')
        video = method6_html_video(username, test_video_id)
        if video:
            print(f'  Description: {video["description"][:80]}')
            print(f'  Plays: {video["stats"]["plays"]:,}')
            print(f'  Likes: {video["stats"]["likes"]:,}')
            print(f'  Comments: {video["stats"]["comments"]:,}')
            print(f'  Shares: {video["stats"]["shares"]:,}')
            print(f'  Hashtags: {video["hashtags"]}')
    
    print('\n=== DONE ===')
