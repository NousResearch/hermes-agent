#!/usr/bin/env python3
"""
MrHermagi Lesson Regeneration v5
Splits the work: hermes -z generates HTML + narration script,
then this script generates audio with kenseivoice/Kokoro directly.

This solves the problem where text_to_speech tool isn't available in CLI mode.
"""

import subprocess
import time
import shutil
import yaml
import os
import sys
from datetime import datetime
from pathlib import Path

CURRICULUM_PATH = Path("/home/kensei/.hermes/profiles/mrhermagi/curriculum.yaml")
BACKUP_CURRICULUM = Path("/home/kensei/.hermes/profiles/mrhermagi/curriculum.yaml.bak-regen")
LOG_PATH = Path("/home/kensei/.hermes/profiles/mrhermagi/regeneration.log")
OUTPUT_BASE = Path("/home/kensei/.hermes/runbooks/mrhermagi")

LESSONS_TO_REGEN = [9, 10, 11, 12]
TIMEOUT_PER_LESSON = 900

# Add KenseiAgent to path for kenseivoice import
sys.path.insert(0, "/home/kensei/repos/KenseiAgent")
os.environ.setdefault("HERMES_HOME", "/home/kensei/.hermes")

LESSON_PROMPT = """You are MrHermagi, Sahil's personal AI/ML teacher. Generate today's lesson.

## TEACHING APPROACH
- Sahil knows: Claude Code, Codex CLI, Cursor, Hermes Agent, APIs, prompting, tokens as a cost concept
- Sahil does NOT know: ML fundamentals, model architecture, training vs inference
- Explain EVERY concept from scratch using analogies tied to his PM/indie-dev experience
- British English, direct, no fluff, NO mermaid diagrams
- Define jargon the moment you use it

## TEACH DOWN TO A BEGINNER
- Lead with a dead-simple analogy BEFORE the technical explanation
- Restate every technical sentence in plain English
- Connect the dots: "because X, that's why Y"
- One new idea per paragraph

## RESEARCH FIRST
Use your skills (arxiv, llm-wiki, market-research, youtube-content) to ground the lesson. Pull at least one primary source and one YouTube video. If web tools fail, check LLM wiki, teach from knowledge, note unverified claims.

## CURRICULUM
Read `/home/kensei/.hermes/profiles/mrhermagi/curriculum.yaml`. Find the day with `status: "next"` — that is today's lesson.

## WHAT TO PRODUCE

### 1. HTML Lesson
Write to: `~/.hermes/runbooks/mrhermagi/YYYY-MM-DD/lesson-N-slug.html`

Use today's date for the directory name (YYYY-MM-DD format).

Required sections (in order):
1. Title: `Week {N}: {Theme} — {Topic} — Day {N} — {Lesson Name}`
2. Learning objective: "By the end you can:"
3. Why this matters — connect to Sahil's tools/products (Plenishd, CoachOS, MatchdayMaestro, Claude Code, Hermes)
4. Concept, built in layers — analogy -> technical -> "In plain English:" restatement
5. Worked example — grounded in Sahil's projects
6. Common pitfalls — 2-3 traps
7. Resources with links — papers, YouTube videos, blog posts, playgrounds, demos
8. Active recall — 2-3 questions in collapsible `<details>` blocks with answer key
9. Recap — 3 bullet takeaways
10. Spiral callback — connect to previous lesson
11. Next up — one line preview

HTML template (mandatory):
- Dark mode: #11100f background, #fbbf24 accent (gold)
- Cards: #1c1a18 / #2c2a28 backgrounds, #34302c borders
- Text: #f5f5f4 body, #a8a29e muted, #f87171 pitfalls
- max-width: 720px centred for mobile
- CSS classes: .analogy, .plain, .pitfall, .work-example, .recap, .callback
- Collapsible: `<details class="detail-block"><summary>Q1: ...</summary><div class="answer">...</div></details>`

### 2. Audio Narration Script
Write script to: `~/.hermes/runbooks/mrhermagi/YYYY-MM-DD/lesson-N-slug.txt`
- No markup, no code symbols, no URLs (describe them)
- Natural, conversational — 2500-3000 chars
- UK spelling
- Start: "Welcome to the AI/ML Foundations Sprint. Lesson {N}: {Title}."
- End: "Next time: {next lesson preview}."

IMPORTANT: Do NOT attempt to generate audio. Just write the narration .txt file. The audio will be generated separately after you finish. Do NOT use text_to_speech, edge-tts, gTTS, or any TTS tool. Just write the .txt script file.

### 3. Update Curriculum
After generating files, update curriculum.yaml:
- Mark today's lesson as `status: delivered`
- Set next lesson to `status: next`

## OUTPUT
Output a brief summary (under 500 chars): which lesson was generated, file paths. Do NOT output MEDIA tags.

GO."""

def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def patch_curriculum_for_day(day_num):
    with open(CURRICULUM_PATH, "r") as f:
        curriculum = yaml.safe_load(f)
    for week in curriculum.get("weeks", []):
        for day in week.get("days", []):
            d = day.get("day")
            if d == day_num:
                day["status"] = "next"
            elif d < day_num:
                day["status"] = "delivered"
            else:
                if "status" in day:
                    del day["status"]
    with open(CURRICULUM_PATH, "w") as f:
        yaml.dump(curriculum, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def find_latest_file(pattern, base=OUTPUT_BASE, min_mtime=None):
    """Find the most recent file matching pattern. If min_mtime is set, only
    return files modified after that timestamp — prevents picking up stale files
    from previous runs when the agent fails to write new ones."""
    files = sorted(base.rglob(pattern), key=lambda p: p.stat().st_mtime)
    if min_mtime is not None:
        files = [f for f in files if f.stat().st_mtime > min_mtime]
    return files[-1] if files else None

def generate_kokoro_audio(txt_path, output_path):
    """Generate audio using kenseivoice/Kokoro directly via Python."""
    try:
        from plugins.tts.kenseivoice import KenseiVoiceProvider
        provider = KenseiVoiceProvider()
        
        # Read narration script
        with open(txt_path, "r") as f:
            text = f.read().strip()
        
        if not text:
            return False, "Empty narration script"
        
        # Truncate to 3000 chars (Kokoro handles this well)
        if len(text) > 3000:
            text = text[:2997] + "..."
        
        # Generate audio with kokoro:bm_george
        result_path = provider.synthesize(
            text=text,
            output_path=str(output_path),
            voice="kokoro:bm_george",
            speed=1.0,
            format="mp3"
        )
        return True, result_path
    except Exception as e:
        return False, str(e)

def verify_lesson(day_num, start_time):
    html_file = find_latest_file(f"lesson-{day_num}-*.html")
    txt_file = find_latest_file(f"lesson-{day_num}-*.txt")
    mp3_file = find_latest_file(f"lesson-{day_num}-*.mp3")
    
    html_new = html_file and html_file.stat().st_mtime > start_time
    txt_new = txt_file and txt_file.stat().st_mtime > start_time
    mp3_exists = mp3_file and mp3_file.stat().st_size > 50000
    
    html_size = html_file.stat().st_size if html_file else 0
    mp3_size = mp3_file.stat().st_size if mp3_file else 0
    
    return (html_new and txt_new and mp3_exists and html_size > 8000), html_size, mp3_size

def delete_old_lesson_files(day_num):
    """Delete all existing files for a lesson day to ensure clean slate."""
    for f in OUTPUT_BASE.rglob(f"lesson-{day_num}-*"):
        try:
            f.unlink()
        except OSError:
            pass

def run_hermes_with_retry(day_num, max_attempts=3):
    """Run hermes -z with 503 retry logic. Returns (success, result, stdout)."""
    for attempt in range(max_attempts):
        try:
            result = subprocess.run(
                ["hermes", "-z", LESSON_PROMPT,
                 "--profile", "mrhermagi",
                 "-m", "glm-5.2",
                 "--provider", "ollama-cloud",
                 "--skills", "lesson-delivery",
                 "--yolo"],
                capture_output=True, text=True, timeout=TIMEOUT_PER_LESSON,
                cwd="/home/kensei/.hermes/profiles/mrhermagi"
            )
            # Check for 503 overload in stderr/stdout
            combined = (result.stderr or "") + (result.stdout or "")
            if "503" in combined and "overloaded" in combined:
                wait = 30 * (attempt + 1)
                log(f"  503 overload detected (attempt {attempt+1}/{max_attempts}), waiting {wait}s...")
                time.sleep(wait)
                continue
            return True, result, result.stdout
        except subprocess.TimeoutExpired:
            log(f"  TIMEOUT (attempt {attempt+1}/{max_attempts})")
            if attempt < max_attempts - 1:
                time.sleep(15)
            continue
        except Exception as e:
            log(f"  ERROR (attempt {attempt+1}/{max_attempts}): {e}")
            if attempt < max_attempts - 1:
                time.sleep(10)
            continue
    return False, None, ""

def main():
    with open(LOG_PATH, "w") as f:
        pass
    
    log("=" * 60)
    log("MR HERMAGI LESSON REGENERATION v5 (hermes -z + Kokoro TTS)")
    log(f"Target: {len(LESSONS_TO_REGEN)} lessons (Days 1-19)")
    log(f"Model: glm-5.2 via ollama-cloud")
    log(f"TTS: kenseivoice/kokoro:bm_george (direct Python call)")
    log("=" * 60)
    
    shutil.copy2(CURRICULUM_PATH, BACKUP_CURRICULUM)
    log("Curriculum backed up")
    
    results = []
    
    for i, day_num in enumerate(LESSONS_TO_REGEN):
        log(f"\n{'='*50}")
        log(f"LESSON {day_num}/19 — STARTING ({i+1}/{len(LESSONS_TO_REGEN)})")
        log(f"{'='*50}")
        
        patch_curriculum_for_day(day_num)
        log(f"Curriculum patched: Day {day_num} = next")
        
        # Clean slate: delete old files for this day to prevent stale detection
        delete_old_lesson_files(day_num)
        log(f"Old files for Day {day_num} deleted")
        
        start_time = time.time()
        
        # Step 1: Run hermes -z with 503 retry
        log(f"Running hermes -z (max {TIMEOUT_PER_LESSON}s, 503 retry)...")
        ok, result, stdout = run_hermes_with_retry(day_num)
        
        if not ok:
            log(f"  All retry attempts failed for Day {day_num}")
            results.append({"day": day_num, "success": False, "html_size": 0, "audio_size": 0, "elapsed": time.time() - start_time})
            continue
        
        elapsed_hermes = time.time() - start_time
        log(f"hermes -z done in {elapsed_hermes:.0f}s (exit={result.returncode})")
        if result.returncode != 0 and result.stderr:
            log(f"  stderr: {result.stderr[:300]}")
        
        # Step 2: Find NEW files only (min_mtime = start_time)
        txt_file = find_latest_file(f"lesson-{day_num}-*.txt", min_mtime=start_time)
        html_file = find_latest_file(f"lesson-{day_num}-*.html", min_mtime=start_time)
        
        if not txt_file or not html_file:
            log(f"  No new files: HTML={'yes' if html_file else 'NO'}, TXT={'yes' if txt_file else 'NO'}")
            results.append({"day": day_num, "success": False, "html_size": 0, "audio_size": 0, "elapsed": time.time() - start_time})
            continue
        
        html_size = html_file.stat().st_size
        txt_size = txt_file.stat().st_size
        log(f"  HTML: {html_size}B, TXT: {txt_size}B")
        
        # Generate audio with Kokoro
        mp3_path = txt_file.with_suffix(".mp3")
        log(f"  Generating Kokoro audio...")
        audio_ok, audio_result = generate_kokoro_audio(txt_file, mp3_path)
        
        if audio_ok:
            audio_size = Path(audio_result).stat().st_size if Path(audio_result).exists() else 0
            log(f"  Kokoro audio: {audio_size}B at {audio_result}")
        else:
            log(f"  Kokoro FAILED: {audio_result}")
            audio_size = 0
        
        elapsed = time.time() - start_time
        success = html_size > 8000 and audio_size > 50000
        
        results.append({
            "day": day_num, "success": success,
            "html_size": html_size, "audio_size": audio_size, "elapsed": elapsed,
        })
        
        if success:
            log(f"  ✓ Day {day_num} OK ({html_size}B HTML, {audio_size}B audio, {elapsed:.0f}s)")
        else:
            log(f"  ✗ Day {day_num} FAIL ({html_size}B HTML, {audio_size}B audio, {elapsed:.0f}s)")
    
    # Restore curriculum from backup (already in correct state: 1-19 delivered, 20 next)
    shutil.copy2(BACKUP_CURRICULUM, CURRICULUM_PATH)
    log("\nCurriculum restored from backup")
    
    # Summary
    log(f"\n{'='*60}")
    log("REGENERATION SUMMARY")
    log(f"{'='*60}")
    ok = sum(1 for r in results if r["success"])
    fail = sum(1 for r in results if not r["success"])
    total = sum(r.get("elapsed", 0) for r in results)
    log(f"OK: {ok}/{len(results)}, FAIL: {fail}, Total: {total/60:.1f}min")
    for r in results:
        s = "OK" if r["success"] else "FAIL"
        log(f"  Day {r['day']:2d}: {s} — HTML {r['html_size']:>6}B, Audio {r['audio_size']:>6}B ({r.get('elapsed',0):.0f}s)")
    if fail:
        log("\nFailed:")
        for r in results:
            if not r["success"]:
                log(f"  Day {r['day']}")
    log("DONE")

if __name__ == "__main__":
    main()