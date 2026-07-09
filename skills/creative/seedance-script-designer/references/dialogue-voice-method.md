# Dialogue and Character Voice Method

Use this reference when a script contains meaningful dialogue, narration, voice-over, chanting, market calls, legal/procedural speech, ritual speech, or character-specific verbal style.

This method is for production writing and AI-video handoff. Do not quote or copy reference books. Apply the reusable principles: dialogue is action, every speaker needs a distinct voice, subtext is stronger than explanation, and silence can carry dramatic meaning.

## 1. Useful Principles

| Principle | Production Use |
|---|---|
| Dialogue is action | Every line should try to achieve something: seduce, block, threaten, comfort, expose, bargain, command, confess, refuse, or delay. |
| Do not explain what the image can show | World rules should be revealed through behavior, transactions, rituals, verdicts, work routines, or consequences. |
| Give each speaker a private lexicon | Character voice should differ by vocabulary, syntax, rhythm, status, profession, fear, desire, and moral position. |
| Avoid on-the-nose dialogue | Do not directly say the subtext when a shorter line, pause, gesture, reaction, or prop interaction can imply it. |
| Use silence and pauses | A held look, stopped hand, failed breath, swallowed line, or no-response can be more useful than another sentence. |
| Keep screen dialogue compact | For video, prioritize short, speakable lines that actors or TTS can perform clearly over literary paragraphs. |
| Separate channels | Keep `台词`, `旁白`, and `状态/音效` separate in 12-column tables and audio sync tables. |

## 2. Dialogue Pass Workflow

Run this pass after story structure and before final video prompts whenever dialogue matters.

1. For each major speaker, define:
   - surface speech style;
   - hidden desire or fear;
   - private lexicon;
   - sentence length and rhythm;
   - what they never say directly.
2. For every key line, identify the action verb:
   - comfort, accuse, trade, test, seduce, command, deny, confess, expose, refuse, bless, curse, bargain, evade.
3. Replace exposition with behavior:
   - market calls reveal economy;
   - official formula reveals power;
   - ritual lines reveal hierarchy;
   - repeated phrases reveal commodification or coercion.
4. Compress lines:
   - remove repeated meaning;
   - prefer a concrete noun or verb over abstract explanation;
   - keep the line speakable in one breath unless ritual or narration requires length.
5. Add subtext notes only in planning tables, not inside visible storyboard images or final video frames.
6. Before final handoff, check that lip-sync lines are short enough for the shot duration.

## 3. Character Voice Bible Format

Use this table when the project has recurring characters or multiple social systems.

| Speaker | Surface Voice | Hidden Action/Subtext | Lexicon | Rhythm | Forbidden Habit |
|---|---|---|---|---|---|
| Character/system name | How the speech sounds | What the line is really doing | Words only this speaker would use | Short/long, broken/smooth, formal/casual | What this speaker should not do |

Example use:

| Speaker | Surface Voice | Hidden Action/Subtext | Lexicon | Rhythm | Forbidden Habit |
|---|---|---|---|---|---|
| Scholar | restrained, polite | hides fear and guilt | mother, exam, dark, debt, lamp | starts evasive, ends very short | explaining his trauma too early |
| Market vendor | warm, practical | turns grief into price | fresh, three nights, oil, worth, regular customer | fast and friendly | sounding like a monster |
| Official | procedural, cold | turns speech into law | testimony, seal, verify, verdict | square and final | explaining morality |

## 4. Line Upgrade Table

Use this table to revise important dialogue without losing production traceability.

| Segment/Shot | Original Function | Current Line | Upgraded Line | Line Action | Why Better |
|---|---|---|---|---|---|
|  |  |  |  |  |  |

Good upgrade patterns:

- Explanation -> transaction: replace "this lamp contains a dead mother's last words" with a vendor call that sells the result.
- Emotion label -> concrete boundary: replace "I am afraid" with "Do not touch the lamp."
- Theme statement -> irreversible choice: replace "the dead should not be used" with a character refusing to light the lamp.
- Long confession -> two short blows: "I did not go in. It was not the fire. I was afraid."

## 5. AI Video Prompt Rules for Dialogue

Add these rules to direct-copy video prompt blocks when dialogue is important:

```text
对白规则：所有对白都必须短、准、有行动。不要解释世界观，不要把人物内心直接讲明。每个说话人要有不同词库和节奏；关键台词前允许停顿、呼吸和反应。画面中不要出现任何字幕、标题、台词文字或气泡；台词只通过口型、声音和表演呈现。
```

For bilingual projects, include only the spoken language line in the direct video prompt unless the user needs translation. Keep translation in the table or notes.

## 6. Dialogue Quality Checklist

| Check | Pass Criteria |
|---|---|
| Action | Each line has a clear purpose beyond "providing information". |
| Speaker identity | Lines could not be swapped between two characters without feeling wrong. |
| Subtext | At least the key lines imply more than they say. |
| Compression | Lines are short enough for the shot duration and performance. |
| Visual support | Dialogue does not repeat what the image already shows. |
| Audio separation | Dialogue, narration, and sound effects remain in separate fields. |
| Final image safety | No subtitles or dialogue text appear inside storyboard/video frames unless explicitly requested. |

## 7. When to Include This in Output

Include a `对白优化与角色声音表` section when:

- the user asks for a full production package;
- the story relies on dialogue, ritual speech, legal speech, market calls, songs, voice-over, or confession;
- multiple factions/systems speak in different registers;
- the user asks to improve lines, tone, character voice, or emotional force.

For narrow visual-only requests, keep this as an internal check and do not add a full section unless dialogue affects the shot.

