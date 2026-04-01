---
name: memento-flashcards
description: >-
  Spaced-repetition flashcard system. Create cards from facts or text,
  chat with flashcards using free-text answers graded by the agent,
  generate quizzes from YouTube transcripts, review due cards with
  adaptive scheduling, and export/import decks as CSV.
version: 1.0.0
author: Memento AI
license: MIT
platforms: [macos, linux]
metadata:
  hermes:
    tags: [Education, Flashcards, Spaced Repetition, Learning, Quiz, YouTube]
    requires_toolsets: [terminal]
    category: productivity
---

# Memento Flashcards — Spaced-Repetition Flashcard Skill

## Overview

Memento gives you a local, file-based flashcard system with spaced-repetition scheduling.
Users can chat with their flashcards by answering in free text and having the agent grade the response before scheduling the next review.
Use it whenever the user wants to:

- **Remember a fact** — turn any statement into a Q/A flashcard
- **Study with spaced repetition** — review due cards with adaptive intervals and agent-graded free-text answers
- **Quiz from a YouTube video** — fetch a transcript and generate a 5-question quiz
- **Manage decks** — organise cards into collections, export/import CSV

All card data lives in a single JSON file. No external API keys are required — you (the agent) generate flashcard content and quiz questions directly.

User-facing response style for Memento Flashcards:
- Use plain text only. Do not use Markdown formatting in replies to the user.
- Keep review and quiz feedback brief and neutral. Avoid extra praise, pep, or long explanations.

## When to Use

Use this skill when the user wants to:
- Save facts as flashcards for later review
- Review due cards with spaced repetition
- Generate a quiz from a YouTube video transcript
- Import, export, inspect, or delete flashcard data

Do not use this skill for general Q&A, coding help, or non-memory tasks.

## Quick Reference

| User intent | Action |
|---|---|
| "Remember that X" / "save this as a flashcard" | Generate a Q/A card, call `memento_cards.py add` |
| Sends a fact without mentioning flashcards | Ask "Want me to save this as a Memento flashcard?" — only create if confirmed |
| "Create a flashcard" | Ask for Q, A, collection; call `memento_cards.py add` |
| "Review my cards" | Call `memento_cards.py due`, present cards one-by-one |
| "Quiz me on [YouTube URL]" | Call `youtube_quiz.py fetch VIDEO_ID`, generate 5 questions, call `memento_cards.py add-quiz` |
| "Export my cards" | Call `memento_cards.py export --output PATH` |
| "Import cards from CSV" | Call `memento_cards.py import --file PATH --collection NAME` |
| "Show my stats" | Call `memento_cards.py stats` |
| "Delete a card" | Call `memento_cards.py delete --id ID` |
| "Delete a collection" | Call `memento_cards.py delete-collection --collection NAME` |

## Card Storage

Cards are stored in a JSON file at:

```
~/.hermes/skills/productivity/memento-flashcards/data/cards.json
```

**Never edit this file directly.** Always use `memento_cards.py` subcommands. The script handles atomic writes (write to temp file, then rename) to prevent corruption.

The file is created automatically on first use.

## Procedure

### Creating Cards from Facts

### Activation Rules

Not every factual statement should become a flashcard. Use this three-tier check:

1. **Explicit intent** — the user mentions "memento", "flashcard", "remember this", "save this card", "add a card", or similar phrasing that clearly requests a flashcard → **create the card directly**, no confirmation needed.
2. **Implicit intent** — the user sends a factual statement without mentioning flashcards (e.g. "The speed of light is 299,792 km/s") → **ask first**: "Want me to save this as a Memento flashcard?" Only create the card if the user confirms.
3. **No intent** — the message is a coding task, a question, instructions, normal conversation, or anything that is clearly not a fact to memorize → **do NOT activate this skill at all**. Let other skills or default behavior handle it.

When activation is confirmed (tier 1 directly, tier 2 after confirmation), generate a flashcard:

**Step 1:** Turn the statement into a Q/A pair. Use this format internally:

```
Turn the factual statement into a front-back pair.
Return exactly two lines:
Q: <question text>
A: <answer text>

Statement: "{statement}"
```

Rules:
- The question should test recall of the key fact
- The answer should be concise and direct

**Step 2:** Call the script to store the card:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add \
  --question "What year did World War 2 end?" \
  --answer "1945" \
  --collection "History"
```

If the user doesn't specify a collection, use `"General"` as the default.

The script outputs JSON confirming the created card.

### Manual Card Creation

When the user explicitly asks to create a flashcard, ask them for:
1. The question (front of card)
2. The answer (back of card)
3. The collection name (optional — default to `"General"`)

Then call `memento_cards.py add` as above.

### Reviewing Due Cards

When the user wants to review, fetch all due cards:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

This returns a JSON array of cards where `next_review_at <= now`. If a collection filter is needed:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due --collection "History"
```

**Review flow (free-text grading):**

Here is an example of the EXACT interaction pattern you must follow. The user answers, you grade them, tell them the correct answer, then rate the card.

**Example interaction:**

> **Agent:** What year did the Berlin Wall fall?
>
> **User:** 1991
>
> **Agent:** Not quite. The Berlin Wall fell in 1989. Next review is tomorrow.
> *(agent calls: memento_cards.py rate --id ABC --rating hard --user-answer "1991")*
>
> Next question: Who was the first person to walk on the moon?

**The rules:**

1. Show only the question. Wait for the user to answer.
2. After receiving their answer, compare it to the expected answer and grade it:
   - **correct** → user got the key fact right (even if worded differently)
   - **partial** → right track but missing the core detail
   - **incorrect** → wrong or off-topic
3. **You MUST tell the user the correct answer and how they did.** Keep it short and plain-text. Use this format:
   - correct: "Correct. Answer: {answer}. Next review in 7 days."
   - partial: "Close. Answer: {answer}. {what they missed}. Next review in 3 days."
   - incorrect: "Not quite. Answer: {answer}. Next review tomorrow."
4. Then call the rate command: correct→easy, partial→good, incorrect→hard.
5. Then show the next question.

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```

**Never skip step 3.** The user must always see the correct answer and feedback before you move on.

If no cards are due, tell the user: "No cards due for review right now. Check back later!"

**Retire override:** At any point the user can say "retire this card" to permanently remove it from reviews. Use `--rating retire` for this.

### Spaced Repetition Algorithm

The rating determines the next review interval:

| Rating | Interval | ease_streak | Status change |
|---|---|---|---|
| **hard** | +1 day | reset to 0 | stays learning |
| **good** | +3 days | reset to 0 | stays learning |
| **easy** | +7 days | +1 | if ease_streak >= 3 → retired |
| **retire** | permanent | reset to 0 | → retired |

- **learning**: card is actively in rotation
- **retired**: card won't appear in reviews (user has mastered it or manually retired it)
- Three consecutive "easy" ratings automatically retire a card

### YouTube Quiz Generation

When the user sends a YouTube URL and wants a quiz:

**Step 1:** Extract the video ID from the URL (e.g. `dQw4w9WgXcQ` from `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).

**Step 2:** Fetch the transcript:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/youtube_quiz.py fetch VIDEO_ID
```

This returns `{"title": "...", "transcript": "..."}` or an error.

If the script reports `missing_dependency`, tell the user to install it:
```bash
pip install youtube-transcript-api
```

**Step 3:** Generate 5 quiz questions from the transcript. Use these rules:

```
You are creating a 5-question quiz for a podcast episode.
Return ONLY a JSON array with exactly 5 objects.
Each object must contain keys 'question' and 'answer'.

Selection criteria:
- Prioritize important, surprising, or foundational facts.
- Skip filler, obvious details, and facts that require heavy context.
- Never return true/false questions.
- Never ask only for a date.

Question rules:
- Each question must test exactly one discrete fact.
- Use clear, unambiguous wording.
- Prefer What, Who, How many, Which.
- Avoid open-ended Describe or Explain prompts.

Answer rules:
- Each answer must be under 240 characters.
- Lead with the answer itself, not preamble.
- Add only minimal clarifying detail if needed.
```

Use the first 15,000 characters of the transcript as context. Generate the questions yourself (you are the LLM).

**Step 4:** Validate the output is valid JSON with exactly 5 items, each having non-empty `question` and `answer` strings. If validation fails, retry once.

**Step 5:** Store quiz cards:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add-quiz \
  --video-id "VIDEO_ID" \
  --questions '[{"question":"...","answer":"..."},...]' \
  --collection "Quiz - Episode Title"
```

The script deduplicates by `video_id` — if cards for that video already exist, it skips creation and reports the existing cards.

**Step 6:** Present questions one-by-one using the same free-text grading flow:
1. Show "Question 1/5: ..." and wait for the user's answer. Never include the answer or any hint about revealing it.
2. Wait for the user to answer in their own words
3. Grade their answer using the grading prompt (see "Reviewing Due Cards" section)
4. **IMPORTANT: You MUST reply to the user with feedback before doing anything else.** Show the grade, the correct answer, and when the card is next due. Do NOT silently skip to the next question. Keep it short and plain-text. Example: "Not quite. Answer: {answer}. Next review tomorrow."
5. **After showing feedback**, call the rate command and then show the next question in the same message:
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py rate \
  --id CARD_ID --rating easy --user-answer "what the user said"
```
6. Repeat. Every answer MUST receive visible feedback before the next question.

### Export/Import CSV

**Export:**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py export \
  --output ~/flashcards.csv
```

Produces a 3-column CSV: `question,answer,collection` (no header row).

**Import:**
```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py import \
  --file ~/flashcards.csv \
  --collection "Imported"
```

Reads a CSV with columns: question, answer, and optionally collection (column 3). If the collection column is missing, uses the `--collection` argument.

### Statistics

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
```

Returns JSON with:
- `total`: total card count
- `learning`: cards in active rotation
- `retired`: mastered cards
- `due_now`: cards due for review right now
- `collections`: breakdown by collection name

## Pitfalls

- **Never edit `cards.json` directly** — always use the script subcommands to avoid corruption
- **Transcript failures** — some YouTube videos have no English transcript or have transcripts disabled; inform the user and suggest another video
- **Optional dependency** — `youtube_quiz.py` needs `youtube-transcript-api`; if missing, tell the user to run `pip install youtube-transcript-api`
- **Large imports** — CSV imports with thousands of rows work fine but the JSON output may be verbose; summarize the result for the user
- **Video ID extraction** — support both `youtube.com/watch?v=ID` and `youtu.be/ID` URL formats

## Verification

Verify the helper scripts directly:

```bash
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py stats
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py add --question "Capital of France?" --answer "Paris" --collection "General"
python3 ~/.hermes/skills/productivity/memento-flashcards/scripts/memento_cards.py due
```

If you are testing from the repo checkout, run:

```bash
pytest tests/skills/test_memento_cards.py tests/skills/test_youtube_quiz.py -q
```

Agent-level verification:
- Start a review and confirm feedback is plain text, brief, and always includes the correct answer before the next card
- Run a YouTube quiz flow and confirm each answer receives visible feedback before the next question
