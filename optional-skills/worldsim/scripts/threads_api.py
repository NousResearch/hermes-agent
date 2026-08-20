#!/usr/bin/env python3
"""
Threads (Meta) Profile & Post Extractor
========================================
Extracts profile data and post content from Threads using:
1. OG meta tags from HTML (no auth required for profiles and public posts)
2. WebFinger for ActivityPub discovery
3. Google-indexed post URLs for recent post discovery

METHODS THAT WORK:
- Profile pages at threads.net/@{user} have OG tags with:
  display_name, username, follower_count, thread_count, bio, profile_pic
- Individual post pages have OG tags with:
  full post text, author info, profile pic
- WebFinger at /.well-known/webfinger gives ActivityPub user IDs
- Post URLs must be known (discoverable via web search)

METHODS THAT DON'T WORK (as of 2025):
- Threads Official API (graph.threads.net) requires OAuth token
- ActivityPub /ap/users/ endpoints return 404 for most users
- No public post listing endpoint exists
"""

import re
import json
import html
import subprocess
import sys

def curl_fetch(url, extra_headers=None, timeout=15):
    """Fetch URL using curl (more reliable than urllib for Threads)."""
    cmd = ['curl', '-s', '-L', '--max-time', str(timeout)]
    if extra_headers:
        for k, v in extra_headers.items():
            cmd.extend(['-H', f'{k}: {v}'])
    cmd.append(url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout+5)
        return result.stdout
    except:
        return None

def extract_og_tags(html_content):
    """Extract OpenGraph, meta description, and Twitter tags from HTML."""
    data = {}
    if not html_content:
        return data
    
    for m in re.finditer(r'property="(og:[^"]+)"\s+content="([^"]*)"', html_content):
        key = m.group(1)
        val = html.unescape(m.group(2))
        if key not in data:
            data[key] = val
    
    for m in re.finditer(r'name="description"\s+content="([^"]*)"', html_content):
        data['description'] = html.unescape(m.group(1))
        break
    
    for m in re.finditer(r'name="(twitter:[^"]+)"\s+content="([^"]*)"', html_content):
        key = m.group(1)
        val = html.unescape(m.group(2))
        if key not in data:
            data[key] = val
    
    return data

def parse_profile_description(desc):
    """Parse '5.5M Followers • 142 Threads • Bio. See the latest...' format."""
    result = {}
    if not desc:
        return result
    
    parts = desc.split(' \u2022 ')  # Split on bullet •
    for part in parts:
        part = part.strip()
        if 'Follower' in part:
            result['followers'] = part.split(' Follower')[0].strip()
        elif part.endswith('Threads') or part.endswith('Thread'):
            result['thread_count'] = part.split(' Thread')[0].strip()
        else:
            bio = re.sub(r'\s*See the latest conversations.*$', '', part)
            if bio:
                result['bio'] = bio
    
    return result

def parse_profile_title(title):
    """Parse 'Display Name (@user) • Threads, Say more' format."""
    result = {}
    if not title:
        return result
    m = re.match(r'^(.+?)\s*\(@(\w+)\)', title)
    if m:
        result['display_name'] = m.group(1).strip()
        result['username'] = m.group(2)
    return result

def get_threads_profile(username):
    """
    Get Threads profile data via OG meta tags.
    Returns dict with: username, display_name, bio, followers, thread_count, 
                       profile_picture_url, url
    """
    username = username.lstrip('@')
    url = f'https://www.threads.net/@{username}'
    
    content = curl_fetch(url)
    tags = extract_og_tags(content)
    
    if not tags or 'og:title' not in tags:
        return {'error': 'Failed to fetch or parse profile', 'username': username}
    
    title = tags.get('og:title', '')
    if title.startswith('Threads') and 'Log in' in title:
        return {'error': 'Profile requires login or not found', 'username': username}
    
    result = {
        'platform': 'threads',
        'url': url,
    }
    
    result.update(parse_profile_title(title))
    result.update(parse_profile_description(tags.get('og:description', '')))
    
    if 'og:image' in tags:
        result['profile_picture_url'] = tags['og:image']
    
    return result

def get_threads_webfinger(username):
    """Get WebFinger data (ActivityPub discovery) for a Threads user."""
    username = username.lstrip('@')
    url = f'https://www.threads.net/.well-known/webfinger?resource=acct:{username}@threads.net'
    
    content = curl_fetch(url, {'Accept': 'application/json'})
    if not content:
        return None
    
    try:
        data = json.loads(content)
        if 'error' in data or 'success' in data and not data['success']:
            return None
        
        result = {'subject': data.get('subject', '')}
        for link in data.get('links', []):
            if link.get('type') == 'application/activity+json':
                result['activitypub_url'] = link['href']
            elif link.get('rel') == 'http://webfinger.net/rel/profile-page':
                result['profile_url'] = link['href']
        return result
    except:
        return None

def get_thread_post(post_url):
    """
    Get content of a specific Threads post via OG tags.
    Returns: text, author, image_url
    """
    content = curl_fetch(post_url)
    tags = extract_og_tags(content)
    
    if not tags or 'og:title' not in tags:
        return {'error': 'Failed to fetch post'}
    
    title = tags.get('og:title', '')
    if 'Log in' in title:
        return {'error': 'Post requires login or not found'}
    
    result = {'url': post_url}
    
    if 'og:description' in tags:
        result['text'] = tags['og:description']
    elif 'description' in tags:
        result['text'] = tags['description']
    
    if 'og:title' in tags:
        # Parse "Display Name (@username) on Threads"
        m = re.match(r'^(.+?)\s*\(@(\w+)\)\s+on\s+Threads', title)
        if m:
            result['author_name'] = m.group(1).strip()
            result['author_username'] = m.group(2)
    
    if 'og:image' in tags:
        result['image_url'] = tags['og:image']
    
    return result

def get_threads_full(username):
    """Get complete profile data combining all methods."""
    profile = get_threads_profile(username)
    wf = get_threads_webfinger(username)
    
    if wf:
        profile['webfinger'] = wf
    
    return profile


# ===== TEST =====
if __name__ == '__main__':
    test_users = sys.argv[1:] if len(sys.argv) > 1 else ['zuck', 'nvidia', 'mosseri']
    
    for user in test_users:
        print(f"\n{'='*60}")
        print(f"  THREADS PROFILE: @{user}")
        print(f"{'='*60}")
        
        data = get_threads_full(user)
        for k, v in sorted(data.items()):
            if k == 'profile_picture_url':
                print(f"  {k}: {str(v)[:80]}...")
            elif k == 'webfinger':
                print(f"  webfinger:")
                for wk, wv in v.items():
                    print(f"    {wk}: {wv}")
            else:
                print(f"  {k}: {v}")
    
    # Test posts
    post_urls = [
        'https://www.threads.net/@zuck/post/DEkvXzbyDS9',
    ]
    
    print(f"\n{'='*60}")
    print(f"  THREADS POSTS")
    print(f"{'='*60}")
    
    for purl in post_urls:
        print(f"\n  URL: {purl}")
        post = get_thread_post(purl)
        for k, v in post.items():
            if k in ('image_url',):
                print(f"  {k}: {str(v)[:80]}...")
            elif k == 'text':
                print(f"  {k}: {v[:300]}{'...' if len(v) > 300 else ''}")
            else:
                print(f"  {k}: {v}")

