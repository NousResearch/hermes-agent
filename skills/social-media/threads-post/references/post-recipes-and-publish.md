# Post types, publish facts + two worked examples

## Conversation-starter post types (pick by niche + goal)
| Type | Why it earns replies | Goal it serves |
|---|---|---|
| Question to the room | invites a direct answer | replies, profile visits |
| Hot take / unpopular opinion | splits the room → discussion | replies, reach |
| Behind-the-scenes | a decision/mistake/process invites "how do you…?" | follows, trust |
| Micro-tip | useful → saves + "what about X?" | follows, saves |
| Short story / lesson | narrative invites "same / here's mine" | replies, follows |
| This-or-that | binary choice is frictionless to answer | replies, velocity |

## Post anatomy (<=500 chars)
```
LINE 1  the hook (the take/question) - the scroll-stopper
BODY    1-2 lines of context or the spicy detail (human voice, no press-release tone)
PROMPT  the reason to reply ("what's your take?" / "am I wrong?") - genuine, not bait
TAG     one primary topic tag (not a pile of hashtags)
MEDIA   optional photo/video (out-engages text-only); inline link if it adds value (no first-comment tax)
```

## WoopSocial Threads publish block (what the API exposes)
```
platform: THREADS
content.text: the post (respect Threads' 500-char limit; validate catches overflow -> CONTENT/DESCRIPTION)
content.media: optional MEDIA_LIBRARY image/video (raw-bytes upload; see tools/integrations/woopsocial.md)
schedule: SCHEDULE_FOR_LATER -> a velocity window (or PUBLISH_NOW)
```
**No `postType`, no `link` field (put links in the text), no poll/voice/GIF** (native-app only). **Single
post only** — the content array is one item, so **no native multi-post chain/thread**. **No update
endpoint** → edit = delete + recreate (only while `NOT_STARTED`). **No reply surface** → the reply work is
human (`community-management`).

## Worked example 1 — hot take (blunt indie-founder voice)
```
POST: "unpopular opinion: most 'AI strategy' decks are a feature list with a logo on top. nobody's shipped anything.
what's one AI thing your team actually uses every day?" TAG: AI. MEDIA: none. SCHEDULE: 8am peak.
REPLY PLAN: answer every reply in the first hour with a follow-up question. WoopSocial publishes the single post.
```

## Worked example 2 — question + photo (warm studio voice)
```
POST: "tax season starts in the chaos drawer, every time. show me your receipt situation - be honest. ours is... a shoebox."
+ photo of the shoebox. TAG: smallbusiness. inline link to a free checklist (rewarded, inline). SCHEDULE: lunchtime.
REPLY PLAN: reply warmly to each, ask one follow-up. Single post via WoopSocial; replies are human.
```

Both: a real reason to reply; human voice (no bait/promo); one topic tag; optional media + inline link;
scheduled to a velocity window; single post via WoopSocial; the reply work is human; metrics from native
analytics (never fabricated).
