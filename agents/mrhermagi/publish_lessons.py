#!/usr/bin/env python3
"""
Publish MrHermagi lessons 1-20 to Discord in chronological order.
Creates weekly forum threads, posts daily comments with homework links,
and replies with HTML + audio file attachments.
"""

import json, os, re, time, yaml
from pathlib import Path
from hermes_tools import send_message

BASE = Path("/home/kensei/.hermes/runbooks/mrhermagi")
CHANNEL_ID = "1507357967731916942"
FORUM_CHANNEL = f"discord:{CHANNEL_ID}"

# Week structure
WEEKS = {
    1: {"theme": "Foundations: Text Models", "days": list(range(1, 8))},
    2: {"theme": "Architecture & All Modalities", "days": list(range(8, 15))},
    3: {"theme": "Running & Comparing Models", "days": list(range(15, 22))},
    4: {"theme": "Advanced: Agents, Fine-tuning, Production", "days": list(range(22, 29))},
}

def get_week(day):
    for w, info in WEEKS.items():
        if day in info["days"]:
            return w, info["theme"]
    return None, None

def find_latest(pattern):
    files = sorted(BASE.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def extract_homework(html_path):
    """Extract homework links from HTML file."""
    content = html_path.read_text(errors='ignore')
    links = re.findall(r'href="(https?://[^"]+)"[^>]*>([^<]+)</a>', content)
    return [{"url": u, "text": t.strip()} for u, t in links[:4]]

def build_daily_message(lesson):
    """Build the Discord daily comment message (under 1500 chars)."""
    day = lesson['day']
    title = lesson['title']
    week = lesson['week']
    week_theme = lesson['week_theme']
    homework = lesson['homework']
    
    # Build homework section (max 3 links)
    hw_lines = []
    for i, h in enumerate(homework[:3]):
        label = ["Watch", "Read", "Try"][i] if i < 3 else "Read"
        text = h['text'][:60]
        hw_lines.append(f"- {label}: {text} → {h['url']}")
    
    msg = f"""Week {week}: {week_theme} — Day {day} — {title}

By the end you can: explain this concept and apply it to your AI tools.

The gist: see the full HTML lesson for the complete breakdown.

Homework:
{chr(10).join(hw_lines)}

Active Recall (reply in thread):
1. What is the key takeaway from this lesson?
2. How does this connect to the previous lesson?

Full lesson + audio attached below."""
    
    if len(msg) > 1500:
        # Trim
        msg = msg[:1497] + "..."
    
    return msg

def publish_lesson(lesson, thread_id):
    """Publish a single lesson as a daily comment + file attachments."""
    day = lesson['day']
    
    # 1. Post daily comment
    msg = build_daily_message(lesson)
    target = f"discord:{CHANNEL_ID}:{thread_id}"
    
    result = send_message(action='send', message=msg, target=target)
    if not result.get('success'):
        print(f"  Day {day}: FAILED to post comment: {result}")
        return False
    
    comment_id = result.get('message_id')
    print(f"  Day {day}: comment posted (msg {comment_id})")
    
    # 2. Post audio attachment as reply
    audio_path = lesson['audio_path']
    if audio_path and os.path.exists(audio_path):
        audio_msg = f"MEDIA:{audio_path}\nAudio narration (Kokoro bm_george)."
        result = send_message(action='send', message=audio_msg, target=target)
        if result.get('success'):
            print(f"  Day {day}: audio attached")
        else:
            print(f"  Day {day}: audio FAILED: {result}")
    
    time.sleep(1)
    
    # 3. Post HTML attachment as reply
    html_path = lesson['html_path']
    if html_path and os.path.exists(html_path):
        html_msg = f"MEDIA:{html_path}\nFull HTML lesson — open in browser."
        result = send_message(action='send', message=html_msg, target=target)
        if result.get('success'):
            print(f"  Day {day}: HTML attached")
        else:
            print(f"  Day {day}: HTML FAILED: {result}")
    
    time.sleep(1)
    return True

def main():
    # Load curriculum for titles
    with open("/home/kensei/.hermes/profiles/mrhermagi/curriculum.yaml") as f:
        cur = yaml.safe_load(f)
    
    titles = {}
    for week in cur.get("weeks", []):
        for d in week.get("days", []):
            titles[d["day"]] = d.get("title", "Unknown")
    
    # Build lesson index for days 1-20
    lessons = []
    for day in range(1, 21):
        html = find_latest(f"lesson-{day}-*.html")
        mp3 = find_latest(f"lesson-{day}-*.mp3")
        ogg = find_latest(f"lesson-{day}-*.ogg")
        audio = mp3 or ogg
        if mp3 and ogg:
            audio = mp3 if mp3.stat().st_mtime > ogg.stat().st_mtime else ogg
        
        w, theme = get_week(day)
        hw = extract_homework(html) if html else []
        
        lessons.append({
            "day": day,
            "title": titles.get(day, "Unknown"),
            "week": w,
            "week_theme": theme,
            "html_path": str(html) if html else None,
            "audio_path": str(audio) if audio else None,
            "homework": hw,
        })
        print(f"Day {day}: {titles.get(day, 'Unknown')} (Week {w}) — HTML {html.stat().st_size if html else 0}B")
    
    # Group by week
    week_threads = {}
    
    # Publish week by week
    for week_num in sorted(WEEKS.keys()):
        week_lessons = [l for l in lessons if l['week'] == week_num]
        if not week_lessons:
            continue
        
        theme = WEEKS[week_num]['theme']
        print(f"\n{'='*50}")
        print(f"WEEK {week_num}: {theme} ({len(week_lessons)} lessons)")
        print(f"{'='*50}")
        
        # Create forum thread
        thread_title = f"Week {week_num}: {theme}"
        result = send_message(
            action='send',
            message=thread_title,
            target=FORUM_CHANNEL
        )
        
        if result.get('success'):
            thread_id = result.get('thread_id')
            print(f"Thread created: {thread_title} (ID: {thread_id})")
            week_threads[week_num] = thread_id
        else:
            print(f"FAILED to create thread: {result}")
            continue
        
        time.sleep(2)
        
        # Publish each lesson in order
        for lesson in week_lessons:
            ok = publish_lesson(lesson, thread_id)
            if not ok:
                print(f"  Day {lesson['day']}: PUBLISHING FAILED")
            time.sleep(2)
    
    # Save thread IDs
    print(f"\n{'='*50}")
    print("PUBLICATION COMPLETE")
    print(f"{'='*50}")
    print(f"Weekly threads created:")
    for w, tid in week_threads.items():
        print(f"  Week {w}: thread {tid}")
    print(f"Total lessons published: {len(lessons)}")

if __name__ == "__main__":
    main()