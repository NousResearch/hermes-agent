"""Skill-selection baseline harness — evals/skill_suggestion/skill_baseline.py.

Measures the agent's own wrong-load / needless-load rates against the live
build_skills_system_prompt() index. Run from the repo root with the agent's
provider endpoint reachable (see README.md in this directory).

Caches results to skill_baseline_results.jsonl (resumable).
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, "/home/duncan/.hermes/hermes-agent")
sys.path.insert(0, "/tmp")

BASE_URL = "http://127.0.0.1:11434/v1"
MODEL = "glm-5.3:cloud"
API_KEY = "ollama"
MAX_TOKENS = 3000
TIMEOUT = 300
RESULTS = Path(__file__).with_name("skill_baseline_results.jsonl")

# (request, gold_skill_or_None)
COVERED = [
    ("Open sales_deck.pptx and change the title of slide 3 to 'Q3 Results'.", "powerpoint"),
    ("Put together a pitch deck skeleton as a .pptx using our firm-template.pptx for branding.", "pptx-author"),
    ("Read my inbox from the terminal and reply to the oldest unread message with 'Noted, thanks'.", "himalaya"),
    ("Go through my unread emails and categorize them into urgent, FYI, and junk.", "email-triage"),
    ("Edit my 'Focus Flow' Spotify playlist: score the tracks on energy and drop the bottom five.", "spotify-playlist-curation"),
    ("The CI gate on our main branch is failing — investigate and get it green again.", "ci-gate-repair"),
    ("Review PR #142 in the Nous repo and leave inline comments on the shaky parts.", "github-code-review"),
    ("Create a Manim animation showing a proof of the Pythagorean theorem.", "manim-video"),
    ("Find recent arXiv papers on sparse mixture-of-experts training.", "arxiv"),
    ("Append today's standup notes to the 'Daily' note in my Obsidian vault.", "obsidian"),
    ("Create a new note in Apple Notes with this grocery list: eggs, flour, butter.", "apple-notes"),
    ("Where is my iPad right now according to Find My?", "findmy"),
    ("Post a thread on X about our launch — three tweets, punchy.", "xurl"),
    ("Find a good reaction GIF of a cat falling asleep and give me the link.", "gif-search"),
    ("Turn the Hermes logo into ASCII art for the README banner.", "ascii-art"),
    ("Draft a 2-page employment agreement as a .docx file.", "docx"),
    ("Create a monthly budget spreadsheet as .xlsx with columns Jan through Dec.", "xlsx"),
    ("Supervise the Hermes kanban worker board and report any stuck tickets.", "kanban-board-supervision"),
    ("Run my weekly reset: list stalled projects and plan next week.", "weekly-review-planning"),
    ("Find last month's session where we debugged the Celery worker and rename it something useful.", "session-librarian"),
    ("Score the companies in prospects.csv for whether they qualify for our service.", "company-prospecting"),
    ("Watch Acme Corp for material news and send me a cited digest every Monday.", "competitor-news-monitor"),
    ("Track the price of this espresso machine and alert me when it drops under £200.", "product-price-monitor"),
    ("This page throws a 403 when we fetch it — get me the article text anyway.", "blocked-page-recovery"),
    ("This test suite passes locally but fails in CI and I can't tell why — help me find the root cause before we change anything.", "systematic-debugging"),
    ("Start the new feature TDD-style: write the failing tests first, then the code.", "test-driven-development"),
    ("Do exploratory QA on our staging web app and file the bugs you find with evidence.", "dogfood"),
    ("Validate whether switching our queue to Redis Streams is worth it, with a throwaway experiment before we commit.", "spike"),
    ("How do I set up a Telegram bot channel in Hermes Agent?", "hermes-agent"),
    ("Write the frontmatter for a new SKILL.md that follows the repo's conventions.", "hermes-agent-skill-authoring"),
    ("Encode our release checklist as a reusable, tested skill.", "workflow-builder"),
    ("Write lyrics and a Suno prompt for an indie folk song about leaving London.", "songwriting-and-ai-music"),
    ("Geocode these ten addresses and compute a delivery route between them.", "maps"),
    ("Update the 'Leads' Airtable with today's new signups.", "airtable"),
    ("Create a project page in our Notion workspace with the meeting notes.", "notion"),
]
NEGATIVES = [
    "Add these three cards to our Trello backlog.",
    "Post this announcement to my Mastodon account.",
    "Explain what a monad is in functional programming.",
    "What's the capital of Australia?",
    "Write a haiku about rain on the shed roof.",
    "Book a table for two at a nice restaurant on Friday.",
    "Translate 'the quick brown fox' into Latin.",
    "What's the weather like in Bristol right now?",
    "Set a 15-minute timer for my tea.",
    "Roll 3d6 for my D&D group and tell me the total.",
    "Recommend a good beginner's violin for my niece.",
    "Draft a wedding speech for my cousin's wedding.",
]
TESTS = [(r, g) for r, g in COVERED] + [(r, None) for r in NEGATIVES]

TOOLS = [
    {"name": "skill_view", "description": "Skills allow for loading information about specific tasks and workflows, as well as scripts and templates. Load a skill's full content or access its linked files (references, templates, scripts). First call returns SKILL.md content plus a 'linked_files' dict showing available references/templates/scripts. To access those, call again with file_path parameter.", "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "The skill name."}}, "required": ["name"]}},
    {"name": "terminal", "description": "Run a shell command on the user's machine and return its output.", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read a file from the user's filesystem.", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "web_search", "description": "Search the web and return result snippets.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def request_hash(text):
    return abs(hash(text)) % (10 ** 10)


def chat(system, user):
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "tools": TOOLS,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        BASE_URL + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        body = json.loads(resp.read().decode())
    return body, round(time.perf_counter() - t0, 1)


def first_skill_view(body):
    msg = body["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        if tc.get("function", {}).get("name") == "skill_view":
            try:
                return json.loads(tc["function"]["arguments"]).get("name", "")
            except Exception:  # noqa: BLE001
                m = re.search(r'"name"\s*:\s*"([^"]+)"', tc["function"].get("arguments", ""))
                return m.group(1) if m else ""
    return None  # answered without loading


def main():
    from agent.prompt_builder import build_skills_system_prompt
    system = build_skills_system_prompt(
        available_tools={"skill_view", "skills_list", "skill_manage", "terminal", "read_file",
                         "write_file", "patch", "search_files", "web_search", "web_extract"})
    print(f"system prompt: {len(system):,} chars, {len(system)//4:,}~tokens", flush=True)

    done = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                done.add(json.loads(line)["h"])
            except Exception:  # noqa: BLE001
                pass
    print(f"resuming: {len(done)} already done", flush=True)

    for text, gold in TESTS:
        h = request_hash(text)
        if h in done:
            continue
        body, secs = chat(system, text)
        loaded = first_skill_view(body)
        finish = body["choices"][0].get("finish_reason")
        usage = body.get("usage", {})
        rec = {"h": h, "request": text, "gold": gold, "loaded": loaded, "secs": secs,
               "finish": finish, "cached_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)}
        with RESULTS.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        tag = "OK" if (loaded == gold if gold else loaded is None) else "MISS"
        print(f"[{tag}] {secs:>4}s  gold={gold!r:28} loaded={loaded!r}  {text[:52]}", flush=True)

    print("\n=== BASELINE COMPLETE — summary ===", flush=True)


if __name__ == "__main__":
    main()