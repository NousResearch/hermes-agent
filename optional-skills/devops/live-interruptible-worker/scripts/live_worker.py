#!/usr/bin/env python3
"""
LiveWorker — real-time interruptible task processor.
Runs in a PTY terminal. Processes items ONE AT A TIME.
Checks for interrupts BETWEEN EVERY HTTP OPERATION via a
cancellation flag set by the stdin reader thread.

Commands (via stdin):
  STOP       — cancel remaining work, print summary
  SKIP [N]   — skip next N items (default 1)
  LIST       — show remaining items & progress
  FOCUS <q>  — keep only items matching <q> (case-insensitive)
  NEXT       — proceed immediately (don't wait for timeout)

Usage:
  cat tasks.json | python3 live_worker.py
  python3 live_worker.py --items "Item 1" "Item 2"
  python3 live_worker.py --type fetch --items "https://example.com"
"""
import json
import sys
import os
import select
import time
import re
import threading
import urllib.request
import urllib.parse
import urllib.error
import socket
from html.parser import HTMLParser

# ─── HTML stripping ───────────────────────────────────────────────

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []
        self.skip_tags = {'script', 'style', 'noscript'}
        self._skip = False
    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip = True
    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self._skip = False
    def handle_data(self, data):
        if not self._skip:
            self.text.append(data.strip())
    def get_text(self):
        return ' '.join(t for t in self.text if t)

def strip_html(html: str) -> str:
    stripper = HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()

# ─── Cancellable fetch with adaptive timeouts ─────────────────────
# Each HTTP request is broken into 1s windows. Between windows,
# the cancellation flag is checked. This means a STOP/SKIP/FOCUS
# takes effect within ~1s even during a slow page load.

SEARXNG_URL = os.environ.get('SEARXNG_URL', 'http://localhost:8080/search')

def cancellable_fetch(url: str, channel: 'CommandChannel',
                      max_timeout: int = 15) -> dict:
    """
    Fetch a URL with cancellation checks every ~1s.
    Uses adaptive timeouts: 1s -> 2s -> 4s -> 8s -> max_timeout.
    Returns partial result {'cancelled': True} if interrupted.
    """
    timeout = 1
    attempts = 0
    user_agent = ('Mozilla/5.0 (X11; Linux x86_64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko)')

    while timeout <= max_timeout:
        if channel.check_cancel():
            return {'url': url, 'cancelled': True}

        attempts += 1
        try:
            req = urllib.request.Request(
                url, headers={'User-Agent': user_agent})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                html = resp.read().decode('utf-8', errors='replace')
                text = strip_html(html)
                shortened = text[:3000] if len(text) > 3000 else text
                result = {
                    'url': url,
                    'content': shortened,
                    'length': len(text),
                    'status': resp.status,
                }
                if channel.check_cancel():
                    result['cancelled'] = True
                return result
        except urllib.error.HTTPError as e:
            return {
                'url': url,
                'error': f'HTTP {e.code}: {e.reason}',
                'status': e.code,
            }
        except urllib.error.URLError as e:
            reason = str(e.reason) if hasattr(e, 'reason') else str(e)
            if 'timed out' in reason.lower():
                print(f"  \u23f3 Timeout after {timeout}s "
                      f"(attempt {attempts}), retrying...")
                sys.stdout.flush()
                timeout = min(timeout * 2, max_timeout)
                continue
            return {'url': url, 'error': reason}
        except (socket.timeout, OSError) as e:
            if 'timed out' in str(e).lower():
                timeout = min(timeout * 2, max_timeout)
                continue
            return {'url': url, 'error': str(e)}

    return {'url': url, 'error': f'Max timeout {max_timeout}s reached'}

# ─── SearXNG search ──────────────────────────────────────────────
def search_web(query: str, limit: int = 5) -> list[dict]:
    """Search via local SearXNG (or configurable endpoint)."""
    params = urllib.parse.urlencode({
        'q': query, 'format': 'json', 'language': 'en'})
    url = f"{SEARXNG_URL}?{params}"
    try:
        req = urllib.request.Request(
            url, headers={'User-Agent': 'LiveWorker/1.0'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
            results = data.get('results', [])
            return [{
                'url': r.get('url', ''),
                'title': r.get('title', ''),
                'snippet': r.get('content', ''),
            } for r in results[:limit]]
    except Exception as e:
        return [{'error': str(e)}]

# ─── Stdin listener with cancellation flag ────────────────────────

class CommandChannel:
    """
    Non-blocking stdin reader + real-time cancellation flag.
    The stdin reader thread sets cancel_current INSTANTLY when
    STOP, SKIP, or FOCUS is received — no poll cycle needed.
    """
    def __init__(self):
        self._buffer = []
        self._lock = threading.Lock()
        self.cancel_current = False
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        """Background thread: read stdin, buffer commands, set cancel flag."""
        try:
            for line in sys.stdin:
                stripped = line.strip()
                cmd = stripped.upper()
                with self._lock:
                    self._buffer.append(stripped)
                    if cmd in ('STOP',) or cmd.startswith(('SKIP', 'FOCUS')):
                        self.cancel_current = True
        except EOFError:
            pass
        except Exception:
            pass

    def poll(self) -> list[str]:
        """Return all pending commands and clear the buffer."""
        with self._lock:
            cmds = list(self._buffer)
            self._buffer.clear()
        return cmds

    def check_cancel(self) -> bool:
        """
        Check if current work should be aborted.
        Called BETWEEN every HTTP operation.
        Thread-safe: stdin reader sets flag, main thread checks it.
        """
        with self._lock:
            if self.cancel_current:
                self.cancel_current = False
                return True
            return False

# ─── Task handlers (interruptible) ───────────────────────────────

def task_search(topic: str, channel: CommandChannel) -> dict:
    """
    Research task: search + fetch top pages.
    Checks channel.check_cancel() BETWEEN EVERY OPERATION.
    Returns partial results if cancelled mid-item.
    """
    results = {
        'topic': topic,
        'search_results': [],
        'pages': [],
        'summary': '',
    }
    print(f"\n{'='*60}")
    print(f"\U0001f50d Searching: {topic}")
    sys.stdout.flush()

    # Step 1: Search web
    search_results = search_web(topic)
    results['search_results'] = search_results
    if channel.check_cancel():
        print("  \u23f9\ufe0f Cancelled mid-search")
        sys.stdout.flush()
        return {**results, '_cancelled': True}

    # Step 2-4: Fetch pages one at a time, checking cancel between each
    for i, sr in enumerate(search_results[:3]):
        url = sr.get('url', '')
        if not url or not url.startswith('http'):
            continue
        print(f"  \U0001f4c4 Fetching [{i+1}/3]: {url[:80]}...")
        sys.stdout.flush()

        page = cancellable_fetch(url, channel)
        results['pages'].append(page)

        if page.get('cancelled'):
            print(f"  \u23f9\ufe0f Cancelled during fetch [{i+1}/3]")
            sys.stdout.flush()
            return {**results, '_cancelled': True}

    # Step 5: Summary
    if results['pages']:
        first = results['pages'][0]
        if 'content' in first:
            content = first['content']
            lines = [l.strip() for l in content.split('\n') if l.strip()]
            meaningful = [
                l for l in lines
                if len(l) > 80 and not l.startswith('http')]
            if meaningful:
                results['summary'] = meaningful[0][:500]

    if channel.check_cancel():
        print("  \u23f9\ufe0f Cancelled during summary")
        sys.stdout.flush()
        return {**results, '_cancelled': True}

    return results


def task_fetch_url(topic: str, channel: CommandChannel) -> dict:
    """Fetch a single URL or search a topic. Interruptible mid-request."""
    if channel.check_cancel():
        return {'_item': topic, '_cancelled': True}

    if topic.startswith('http'):
        print(f"  \U0001f4c4 Fetching: {topic[:80]}...")
        sys.stdout.flush()
        result = cancellable_fetch(topic, channel)
        result['_item'] = topic
        return result
    else:
        print(f"  \U0001f50d Searching: {topic}")
        sys.stdout.flush()
        sr = search_web(topic)
        if channel.check_cancel():
            return {'_item': topic, '_cancelled': True, 'results': sr}
        return {'_item': topic, 'results': sr}


TASK_HANDLERS = {
    'search': task_search,
    'fetch': task_fetch_url,
}

# ─── Command handling ─────────────────────────────────────────────

def handle_command(cmd: str, remaining: list,
                   completed: list, total: int) -> str:
    """Handle a user command. Returns an action key."""
    cmd = cmd.strip().upper()
    if not cmd or cmd == 'NEXT':
        print("  \u23e9 Continuing...")
        sys.stdout.flush()
        return 'next'
    elif cmd == 'STOP':
        print(f"  \U0001f6d1 STOP received — "
              f"cancelling {len(remaining)} remaining items")
        sys.stdout.flush()
        return 'stop'
    elif cmd == 'LIST':
        print(f"  \U0001f4cb Progress: {len(completed)}/{total} done")
        print(f"     Remaining ({len(remaining)}):")
        for i, r in enumerate(remaining[:10]):
            print(f"     {i+1}. {r[:60]}")
        if len(remaining) > 10:
            print(f"     ... and {len(remaining)-10} more")
        sys.stdout.flush()
        return 'list'
    elif cmd.startswith('SKIP'):
        parts = cmd.split()
        if len(parts) > 1 and parts[1].isdigit():
            n = int(parts[1])
            actual = min(n, len(remaining))
            print(f"  \u23ed\ufe0f Skipped {actual} items "
                  f"({[r[:40] for r in remaining[:actual]]})")
            del remaining[:actual]
        else:
            if remaining:
                skipped = remaining.pop(0)
                print(f"  \u23ed\ufe0f Skipped: {skipped[:50]}")
        return 'skip'
    elif cmd.startswith('FOCUS'):
        query = cmd[5:].strip()
        if query:
            before = len(remaining)
            remaining[:] = [
                r for r in remaining if query.lower() in r.lower()]
            filtered = before - len(remaining)
            print(f"  \U0001f3af Focus on '{query}' — "
                  f"kept {len(remaining)}/{before} "
                  f"(filtered {filtered})")
            sys.stdout.flush()
        return 'focus'
    else:
        print(f"  \u2753 Unknown command: {cmd}")
        print("     Available: STOP | SKIP [N] | LIST "
              "| FOCUS <q> | NEXT (or blank)")
        sys.stdout.flush()
        return 'unknown'


# ─── Main loop ───────────────────────────────────────────────────

def run(tasks: list, task_type: str = 'search'):
    handler = TASK_HANDLERS.get(task_type, task_search)
    total = len(tasks)
    completed = []
    remaining = list(tasks)
    skipped = 0
    interrupted = False

    print(f"\n\U0001f680 LiveWorker started — {total} items, type={task_type}")
    print("   Commands: STOP | SKIP [N] | LIST | FOCUS <q> | NEXT")
    print("   (cancellation takes effect between HTTP "
          "operations — typically 1-15s)")
    print(f"{'='*60}")
    sys.stdout.flush()

    channel = CommandChannel()
    time.sleep(0.2)  # give stdin thread a moment to start

    while remaining and not interrupted:
        # Check for commands before popping next item
        cmds = channel.poll()
        for cmd in cmds:
            handled = handle_command(cmd, remaining, completed, total)
            if handled in ('stop',):
                interrupted = True
                break
            elif handled in ('skip',):
                skipped += 1
                break

        if interrupted:
            break

        # Skip to next iteration if remaining was emptied by skip
        if not remaining:
            break

        # Pop and process the next item
        item = remaining.pop(0)
        print(f"\n--- [{len(completed)+1}/{total}] "
              f"Processing: {item[:100]} ---")
        sys.stdout.flush()

        try:
            result = handler(item, channel)
            result['_item'] = item
            is_cancelled = result.pop('_cancelled', False)
            if is_cancelled:
                result['_status'] = 'cancelled'
                print("  \u23f9\ufe0f Item cancelled mid-process "
                      "(partial results saved)")
                sys.stdout.flush()
            else:
                result['_status'] = 'completed'
                if 'summary' in result and result['summary']:
                    print(f"  \u2713 Key finding: "
                          f"{result['summary'][:200]}")
                if 'search_results' in result:
                    for sr in result['search_results'][:3]:
                        print(f"  \u2022 {sr.get('title', '?')[:70]}")
                print("  \u2713 Done.")
                sys.stdout.flush()
            completed.append(result)
        except Exception as e:
            print(f"  \u2717 Error: {e}")
            completed.append({
                '_item': item, '_status': 'error', 'error': str(e)})
            sys.stdout.flush()

        # After each item: wait for commands with 0.1s poll interval
        if not interrupted:
            print(f"\n\u23f3 Waiting for command — send STOP | SKIP [N] "
                  f"| LIST | FOCUS <q> | NEXT")
            print(f"   Progress: {len(completed)}/{total} done, "
                  f"{len(remaining)} remaining")
            print("   (will auto-continue in 10s, "
                  "or send NEXT to continue now)")
            sys.stdout.flush()

            waited = 0
            auto_continue = False
            while waited < 10.0:
                cmds = channel.poll()
                if cmds:
                    for cmd in cmds:
                        handled = handle_command(
                            cmd, remaining, completed, total)
                        if handled == 'stop':
                            interrupted = True
                            auto_continue = True
                            break
                        elif handled in ('skip',):
                            skipped += 1
                            auto_continue = True
                            break
                        elif handled == 'next':
                            auto_continue = True
                            break
                        elif handled == 'focus':
                            auto_continue = False
                            break
                    if interrupted:
                        break
                if auto_continue:
                    break
                if waited > 0 and waited % 2 == 0 and waited < 10:
                    remaining_secs = 10 - waited
                    if remaining_secs > 0 and remaining_secs % 2 == 0:
                        print(f"   Continuing in {int(remaining_secs)}s...")
                        sys.stdout.flush()
                time.sleep(0.1)
                waited = round(waited + 0.1, 1)

    # Final summary
    completed_count = len(
        [c for c in completed if c.get('_status') == 'completed'])
    cancelled_count = len(
        [c for c in completed if c.get('_status') == 'cancelled'])
    error_count = len(
        [c for c in completed if c.get('_status') == 'error'])

    print(f"\n{'='*60}")
    print("\U0001f3c1 LiveWorker finished!")
    print(f"   Completed: {completed_count}")
    print(f"   Cancelled: {cancelled_count}")
    print(f"   Errors:    {error_count}")
    print(f"   Skipped:   {skipped}")
    if interrupted:
        print(f"   Remaining: {len(remaining)} (cancelled by STOP)")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Write final report
    report = {
        'task_type': task_type,
        'total': total,
        'completed': completed_count,
        'cancelled': cancelled_count,
        'errors': error_count,
        'skipped': skipped,
        'remaining': len(remaining),
        'interrupted': interrupted,
        'items': completed,
    }
    cwd = os.getcwd()
    report_path = os.path.join(
        cwd, f"liveworker_report_{int(time.time())}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n\U0001f4c4 Report saved: {report_path}")
    sys.stdout.flush()

    return report


# ─── Entry ────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='LiveWorker — real-time interruptible task processor')
    parser.add_argument(
        '--type', default='search', choices=['search', 'fetch'],
        help='Task type (default: search)')
    parser.add_argument(
        '--items', nargs='*',
        help='Items to process (space-separated)')
    parser.add_argument(
        '--file', help='JSON file with items array')

    args = parser.parse_args()

    tasks = []

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
            if isinstance(data, list):
                tasks = data
            elif isinstance(data, dict) and 'items' in data:
                tasks = data['items']
    elif args.items:
        tasks = args.items
    else:
        # Read from stdin (pipe)
        if not sys.stdin.isatty():
            raw = sys.stdin.read().strip()
            if raw:
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        tasks = parsed
                    elif isinstance(parsed, dict) and 'items' in parsed:
                        tasks = parsed['items']
                except json.JSONDecodeError:
                    tasks = [l.strip()
                             for l in raw.split('\n') if l.strip()]

    if not tasks:
        print("No tasks provided. Pass --items, --file,"
              "or pipe JSON to stdin.")
        print("Examples:")
        print("  python3 live_worker.py --items "
              "'Plausible Analytics' 'Umami' 'Matomo'")
        print("  echo '[\"Item 1\",\"Item 2\"]' | "
              "python3 live_worker.py")
        sys.exit(1)

    run(tasks, task_type=args.type)
