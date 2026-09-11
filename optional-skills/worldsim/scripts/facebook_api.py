#!/usr/bin/env python3
"""
Facebook Page/Profile Data Extractor
Uses multiple techniques to extract public Facebook data without authentication:
1. Googlebot UA for OG meta tags (name, description, likes, talking_about, bio, og:image)
2. Graph API /picture endpoint for profile photos (pages only)
3. Page Plugin embed for follower counts and page IDs
"""

import subprocess
import json
import re
import html
import sys

GOOGLEBOT_UA = 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)'

def curl_get(url, ua=None):
    """Fetch URL with curl"""
    cmd = ['curl', '-s', '-L', '--max-time', '15']
    if ua:
        cmd += ['-H', f'User-Agent: {ua}']
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    return result.stdout

def extract_og_data(username):
    """Extract OG meta tags using Googlebot UA"""
    content = curl_get(f'https://www.facebook.com/{username}', ua=GOOGLEBOT_UA)
    
    data = {}
    
    # Extract OG tags
    og_title = re.search(r'og:title"\s*content="([^"]*)"', content)
    if og_title:
        data['name'] = html.unescape(og_title.group(1))
    
    og_desc = re.search(r'og:description"\s*content="([^"]*)"', content)
    if og_desc:
        desc = html.unescape(og_desc.group(1))
        data['raw_description'] = desc
        
        # Parse likes count
        likes_match = re.search(r'([\d,]+)\s+likes?', desc)
        if likes_match:
            data['likes'] = likes_match.group(1)
        
        # Parse talking about
        talking_match = re.search(r'([\d,]+)\s+talking about this', desc)
        if talking_match:
            data['talking_about'] = talking_match.group(1)
        
        # Extract bio (text after the "talking about this." part)
        bio_match = re.search(r'talking about this\.\s*(.+)', desc)
        if bio_match:
            data['bio'] = bio_match.group(1)
    
    og_image = re.search(r'og:image"\s*content="([^"]*)"', content)
    if og_image:
        data['og_image'] = html.unescape(og_image.group(1))
    
    return data

def extract_plugin_data(username):
    """Extract data from Page Plugin embed"""
    content = curl_get(f'https://www.facebook.com/plugins/page.php?href=https://www.facebook.com/{username}&tabs=timeline&width=500&height=600')
    
    data = {}
    
    # Page name from title attribute
    name_match = re.search(r'class="_1drp _5lv6" title="([^"]*)"', content)
    if name_match:
        data['plugin_name'] = html.unescape(name_match.group(1))
    
    # Follower count
    followers_match = re.search(r'([\d,]+)\s+followers', content)
    if followers_match:
        data['followers'] = followers_match.group(1)
    
    # Page ID
    pageid_match = re.search(r'"pageID":"(\d+)"', content)
    if pageid_match:
        data['page_id'] = pageid_match.group(1)
    
    return data

def extract_profile_picture(username):
    """Get profile picture via Graph API"""
    content = curl_get(f'https://graph.facebook.com/v19.0/{username}/picture?redirect=false&width=400&height=400')
    try:
        d = json.loads(content)
        if 'data' in d and not d['data'].get('is_silhouette', True):
            return d['data']['url']
    except:
        pass
    return None

def get_facebook_data(username):
    """Combine all extraction methods"""
    result = {'username': username}
    
    # Method 1: OG tags (best for bio, likes, talking_about)
    og = extract_og_data(username)
    result.update(og)
    
    # Method 2: Plugin (best for followers, page_id)
    plugin = extract_plugin_data(username)
    result.update(plugin)
    
    # Method 3: Graph API picture (pages only)
    pic = extract_profile_picture(username)
    if pic:
        result['profile_picture'] = pic
    
    # Also try by page_id for picture if username didn't work
    if not pic and 'page_id' in result:
        pic2 = extract_profile_picture(result['page_id'])
        if pic2:
            result['profile_picture'] = pic2
    
    return result

if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['zuck', 'NVIDIA', 'Meta', 'CocaCola']
    
    for target in targets:
        print(f"{'='*60}")
        print(f"Facebook Profile: {target}")
        print(f"{'='*60}")
        data = get_facebook_data(target)
        for k, v in data.items():
            if k == 'raw_description':
                continue  # Skip raw, we show parsed fields
            val = str(v)
            if len(val) > 120:
                val = val[:120] + '...'
            print(f"  {k}: {val}")
        print()

