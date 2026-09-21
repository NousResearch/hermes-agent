#!/usr/bin/env python3
"""
Publish MrHermagi lessons to Discord via hermes send-message CLI.
Uses subprocess to call: hermes send --target <target> --message <msg>
"""

import subprocess, json, os, re, time, yaml, sys
from pathlib import Path

BASE = Path("/home/kensei/.hermes/runbooks/mrhermagi")
CHANNEL = "1507357967731916942"

WEEKS = {
    1: {"theme": "Foundations: Text Models", "days": range(1, 8)},
    2: {"theme": "Architecture & All Modalities", "days": range(8, 15)},
    3: {"theme": "Running & Comparing Models", "days": range(15, 22)},
}

DAY_TITLES = {
    1: "What Is a Language Model?",
    2: "Tokens: Your AI's Alphabet",
    3: "Token-Maxxing & Tokenisation Deep-Dive",
    4: "Parameters: What 7B vs 70B Means",
    5: "Prompt Processing & Generation Loop",
    6: "Sampling & Temperature",
    7: "Model Families Tour + How to Pick",
    8: "The Transformer — How Language Models Process Text",
    9: "Attention: How Models Focus",
    10: "Dense vs MoE: Why DeepSeek is Different",
    11: "Image Models: How AI Sees",
    12: "Voice & Audio Models: How AI Hears",
    13: "Video Models: How AI Generates Motion",
    14: "Multimodal Models: How Everything Connects",
    15: "Quantisation: Making Big Models Fit",
    16: "Context Windows: Lost in the Middle",
    17: "Benchmarks: What Scores Mean",
    18: "Model Evaluation Workshop",
    19: "Token-Maxxing Strategies",
    20: "Model Comparison Framework",
}

def find_latest(pattern):
    files = sorted(BASE.rglob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None

def extract_homework(html_path):
    content = html_path.read_text(errors='ignore')
    links = re.findall(r'href="(https?://[^"]+)"[^>]*>([^<]+)</a>', content)
    return [{"url": u, "text": t.strip()[:60]} for u, t in links[:3]]

def send(target, message):
    """Send a message via hermes send CLI."""
    result = subprocess.run(
        ["hermes", "send", "-t", target, "--json", message],
        capture_output=True, text=True, timeout=60,
        cwd="/home/kensei"
    )
    try:
        return json.loads(result.stdout), result.returncode
    except:
        return None, result.returncode

def send_lesson(day, week, week_theme, thread_id):
    """Send a single lesson: comment + audio + HTML."""
    html = find_latest(f"lesson-{day}-*.html")
    mp3 = find_latest(f"lesson-{day}-*.mp3")
    ogg = find_latest(f"lesson-{day}-*.ogg")
    audio = mp3 or ogg
    if mp3 and ogg:
        audio = mp3 if mp3.stat().st_mtime > ogg.stat().st_mtime else ogg
    
    hw = extract_homework(html) if html else []
    title = DAY_TITLES.get(day, f"Day {day}")
    
    # Build homework lines
    labels = ["Watch", "Read", "Try"]
    hw_lines = []
    for i, h in enumerate(hw[:3]):
        label = labels[i] if i < 3 else "Read"
        hw_lines.append(f"- {label}: {h['text']} → {h['url']}")
    
    # Build message (under 1500 chars)
    msg = f"Week {week}: {week_theme} — Day {day} — {title}\n\n"
    msg += f"By the end you can: explain this concept and apply it to your daily AI tools.\n\n"
    msg += f"The gist: see full HTML lesson for the complete breakdown.\n\n"
    msg += f"Homework:\n{chr(10).join(hw_lines)}\n\n"
    msg += f"Active Recall (reply in thread):\n1. What is the key takeaway?\n2. How does this connect to the previous lesson?\n\n"
    msg += f"Full lesson + audio attached below."
    
    if len(msg) > 1500:
        msg = msg[:1497] + "..."
    
    target = f"discord:{CHANNEL}:{thread_id}"
    
    # 1. Send daily comment
    data, code = send(target, msg)
    print(f"  Day {day}: comment sent (ok={data is not None})", flush=True)
    time.sleep(1)
    
    # 2. Send audio
    if audio and audio.exists():
        audio_msg = f"MEDIA:{audio}\nAudio narration (Kokoro bm_george)."
        data, code = send(target, audio_msg)
        print(f"  Day {day}: audio sent (ok={data is not None})", flush=True)
        time.sleep(1)
    
    # 3. Send HTML
    if html and html.exists():
        html_msg = f"MEDIA:{html}\nFull HTML lesson — open in browser."
        data, code = send(target, html_msg)
        print(f"  Day {day}: HTML sent (ok={data is not None})", flush=True)
        time.sleep(1)

def main():
    log = open("/tmp/publish_lessons.log", "w")
    
    for week_num in sorted(WEEKS.keys()):
        theme = WEEKS[week_num]["theme"]
        days = list(WEEKS[week_num]["days"])
        
        print(f"\n{'='*50}")
        print(f"WEEK {week_num}: {theme} ({len(days)} lessons)")
        print(f"{'='*50}", flush=True)
        
        # Create forum thread
        thread_msg = f"Week {week_num}: {theme}\n\nDaily AI/ML lessons — Week {week_num} covers {theme}. Each day has a summary with homework links, plus full HTML lesson and audio narration attached."
        data, code = send(f"discord:{CHANNEL}", thread_msg)
        
        thread_id = None
        if data:
            thread_id = data.get('thread_id', data.get('message_id'))
        
        if not thread_id:
            print(f"FAILED to get thread ID! data: {data}", flush=True)
            continue
        
        print(f"Thread ID: {thread_id}", flush=True)
        time.sleep(2)
        
        # Send each lesson
        for day in days:
            if day > 20:
                continue
            send_lesson(day, week_num, theme, thread_id)
            time.sleep(2)
    
    print(f"\n{'='*50}")
    print("PUBLICATION COMPLETE")
    print(f"{'='*50}", flush=True)
    log.close()

if __name__ == "__main__":
    main()