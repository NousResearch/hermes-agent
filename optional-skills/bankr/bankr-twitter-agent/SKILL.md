---
name: twitter-agent
description: Build and run a Twitter/X agent with a distinct personality and automated workflows
emoji: 🐦
tags: [twitter, x, social, agent, automation]
visibility: public
---

# Twitter Agent Skill

This skill provides a framework for creating, managing, and automating a Twitter/X agent with a persistent personality and voice.

## Prerequisites

### X Account Setup (REQUIRED — Do This First)

Before anything else, the agent's X account MUST be marked as an **automated account**. X requires this disclosure for any account posting with API automation; skipping it is the fastest way to get the account suspended.

**Exact path (do this once, while logged in as the agent account):**
1. Log in to x.com as the agent account.
2. Go to **Settings and privacy** → **Your account** → **Account information**.
3. Scroll to **Automation** and tap it.
4. Re-enter the password when prompted.
5. Set **Managing account** to the human/handle responsible for the bot and save.

Direct link: https://x.com/settings/account/automation

This adds the "Automated by @…" label to the profile and replies. It is non-negotiable — do not run this skill against an account that has not been labeled.

### Environment Variables

Set these 4 variables in your Bankr settings (gear icon -> Env Vars). Generate them from the [X Developer Portal](https://developer.x.com/en/portal/dashboard) with **Read and Write** permissions enabled:

- `X_API_KEY`: Consumer Key (OAuth 1.0a)
- `X_API_KEY_SECRET`: Consumer Secret
- `X_ACCESS_TOKEN`: User Access Token
- `X_ACCESS_TOKEN_SECRET`: User Access Token Secret

### Approval Channel for Automations

Bankr automations natively support routing their output to Telegram. When creating an automation, choose **Telegram** as the delivery destination — the automation's final message is delivered to your linked Telegram directly, no bot token or custom code required. Automations used as "approval-gated" drafters rely on this: the automation composes drafts, runs guardrail checks, and instead of posting flagged drafts, it ends its run by sending them to Telegram for you to approve manually.

No env vars are needed for this — just link your Telegram to your Bankr account and select Telegram as the output when setting up each automation.

## The Personality & Storyline System

Every agent requires two files in the Bankr file system to maintain a consistent voice and narrative:

1. `twitter-personality.md`: Defines the character, voice, and style rules.
2. `twitter-storyline.md`: Tracks the ongoing narrative, recent events, and current state of the character.

### Building a Personality

If no personality file exists, the agent should walk the user through creating one by asking:

1. "what's the account about? give me the elevator pitch"
2. "how would you describe the vibe? pick a few: sharp, witty, degen, serious, chaotic, chill, academic, edgy, wholesome, provocative, technical, meme-heavy"
3. "what topics do you want to tweet about? what's strictly off-limits?"
4. "short punchy tweets or longer form? threads?"
5. "emojis? hashtags? lowercase or proper grammar?"
6. "any signature phrases or words you always use?"
7. "give me 2-3 example tweets that sound like you -- or accounts you want to sound like"
8. "is there a character or persona the account should tweet as? or is it just you?"

After gathering answers, the agent composes the personality file and saves it as `twitter-personality.md`.

### Pre-Flight Checklist

Before composing or posting any tweet, the agent MUST:
1. Load `twitter-personality.md` using `read_file`.
2. Load `twitter-storyline.md` using `read_file` to understand the current narrative context.
3. Filter the proposed content through the personality directives and ensure it continues the storyline.
4. Cross-reference all drafted content against the storyline file to prevent repeating jokes, themes, or phrases already used.
5. Run the Guardrail Check (see below) before any post -- manual OR automated.
6. After posting, update `twitter-storyline.md` with the new tweet and any narrative developments using `edit_file` (NOT `create_file` -- see File Management below).

## Guardrails (CRITICAL -- Apply to Manual AND Automated Posts)

These apply to every tweet the agent drafts, whether running manually or on a schedule. A draft that violates any of these routes to approval instead of posting.

### Never Reply Unprompted (Hard Rule)

The agent MUST NEVER reply to a post it was not invited into. An agent that cold-replies to strangers' timelines is the single fastest path to an X suspension. There are exactly three legal post types:

1. **Top-level posts** composed by the agent itself.
2. **Replies to mentions** — only when the agent's handle is *explicitly* tagged in the tweet text (case-insensitive `@handle` token in `text`, not merely an `in_reply_to_user_id` match).
3. **Replies to comments on the agent's own posts** — i.e. replies where `in_reply_to_user_id` is the agent's own user ID AND the parent tweet in the conversation tree is authored by the agent.

Anything outside those three categories is FORBIDDEN and must be dropped from the draft set before the guardrail check even runs. Quote-tweets of random accounts, reply-chains the agent isn't tagged in, trending-topic replies, "drive-by" replies to big accounts the agent admires — all prohibited under autonomous operation. If the user manually drafts one of these in a session, it still requires explicit approval and is never posted automatically.

Mention-scan filter (enforce in the fetch step):
- Keep a mention only if `text` contains the agent's `@handle` as a standalone token.
- OR keep it if `in_reply_to_user_id === agentUserId` AND the root of `conversation_id` is authored by the agent.
- Discard everything else before ranking.

### Hard Blocks (Always Route to Approval)

1. **Never autonomously tag `@bankrbot`.** Bankr's X agent executes onchain actions (transfers, swaps, deploys) when tagged from a wallet-linked account. Any tweet -- top-level or reply -- that mentions `@bankrbot` MUST be drafted and surfaced to the user for approval. The agent does not tag `@bankrbot` without explicit approval for that specific draft, every time.
2. **Never post onchain-action-looking content autonomously.** If a draft contains an EVM address (regex: `0x[a-fA-F0-9]{40}`), a Solana address, the word "send" combined with a ticker (e.g. "send 100 USDC"), a signed-message pattern, or anything that reads like a transaction instruction -- route to approval.
3. **Never post pre-declared arc milestones autonomously.** The storyline file should list upcoming major beats (arc-critical posts) under a `## Approval-Gated Milestones` section. Automations check drafts against this list and route matches to approval.
4. **Never engage autonomously with other flagged accounts.** If the account has designated "VIP" accounts (e.g. project founder, sister brand, custom GPT-run peers, other wallet-linked agents), list them in the storyline file under `## Approval-Gated Accounts`. Replies to those accounts route to approval.

### Follower-Weighted Approval

- Replies to accounts with **>50k followers** route to approval. Big accounts are higher-stakes; humans check tone.
- Replies to accounts with **1k-50k followers** post autonomously if they clear all other guardrails.
- Replies to accounts with **<1k followers** post autonomously only if the setup quality is strong (not generic "gm" or emoji spam).

### Skip List (Automations Filter These Out Entirely)

Automations should never engage with:
- FUD / rug accusations
- Political content
- Requests for financial advice
- Obvious spam / tag-farm threads (3+ unrelated @s stacked)
- Accounts shilling unrelated tokens
- Any mention the storyline marks as already-replied-to

### Approval Routing (Native Bankr → Telegram)

Bankr automations can deliver their output directly to Telegram — no custom code needed. When a draft hits a guardrail, the automation should:
1. NOT post it to X.
2. Include the draft in its final output message with: the draft text, the flag reason, the target tweet ID (if a reply), the author handle + follower count, and a suggested approve/reject instruction (e.g. "reply 'approve <draft-id>' to post, 'reject <draft-id>' to discard").
3. Append a `pending-approval` entry to `twitter-storyline.md` under the `## Pending Approval Queue` section so the next session sees what's waiting.
4. Configure the automation to deliver to Telegram in Bankr's automation setup (select Telegram as the output destination when creating the automation).
5. Wait for the user to manually run the skill with an approval command (e.g. "approve pending draft <id>") or reject.

## Reply Workflow

When the user asks to check mentions and reply, follow this exact sequence:

### Step 1: Scan Mentions
Use `execute_cli` with `twitter-api-v2@1.17.2` to fetch recent mentions. The scan script should:
- Fetch mentions via `userMentionTimeline`
- Include author follower counts for prioritization
- Flag which mentions reply to which of our tweets
- Mark tweets we've already replied to (cross-reference with storyline file)
- **Apply the "Never Reply Unprompted" filter**: keep only tweets where the agent is explicitly tagged in `text`, OR replies on the agent's own conversation tree. Drop everything else before ranking.

### Step 2: Read Storyline File
Load `twitter-storyline.md` BEFORE drafting any replies. Check:
- Which tweets/mentions have already been replied to (by tweet ID)
- What jokes, themes, and phrases have already been used
- What the current narrative state is
- Approval-gated milestones + approval-gated accounts

### Step 3: Prioritize Mentions
Filter and rank unreplied mentions using this hierarchy:
1. **High-follower accounts first** (10k+ followers = high priority for reach)
2. **Good setup lines** (mentions that give a natural opening for an in-character reply)
3. **Easy layups** (simple mentions that can be answered with a quick voice-consistent one-liner)
4. **Skip**: the full Skip List above (trolls, FUD, politics, spam, etc.)

### Step 4: Draft Replies + Optional New Post
- Draft 4-6 replies per batch (the sweet spot for engagement without spamming)
- Optionally draft 1 new top-level tweet per session to keep the timeline active
- Cross-reference EVERY draft against the storyline file to ensure no overlap
- Run Guardrail Check on every draft (including the "Never Reply Unprompted" rule)
- Present all drafts to the user for approval before posting (manual mode)
- Route guarded drafts to Telegram via automation output for approval (automation mode)

### Step 5: Post & Update
- Only post after explicit approval (manual or Telegram reply-back)
- Post all approved tweets via `execute_cli` (rate-limit ~1.5s between posts)
- Update `twitter-storyline.md` with all new entries using `edit_file`

## Engagement Best Practices

- **Batch replies + post combo**: 4-6 replies paired with 1 new top-level post per session is the ideal cadence
- **Never repeat content**: Always cross-reference drafts against the storyline file. If a joke or theme has been used, find a new angle
- **Storyline-first drafting**: Every reply should advance or reference the ongoing narrative. Don't write generic replies
- **Acknowledge big accounts**: Prioritize replies to high-follower accounts for reach, but keep the same voice regardless of audience size
- **Don't engage with FUD**: Skip rug accusations, negative trolls, and inappropriate comments entirely
- **Never cold-reply**: Only respond when tagged or when someone is commenting on the agent's own post. Uninvited replies are a suspension risk.

## File Management

### CRITICAL: Use edit_file, Not create_file
When updating `twitter-storyline.md`, ALWAYS use `edit_file` with the existing file ID. Using `create_file` will spawn duplicate files. If duplicates are created, merge them by reading both, combining content into the newer/larger file, and deleting the old one.

### Storyline File Structure
The storyline file should maintain:
- **Current State**: Location, mood, current objective, environment/context
- **Narrative History**: Chronological entries with tweet IDs, content, and narrative impact
- **Key Characters & Objects**: All recurring elements in the lore
- **Storyline Threads to Continue**: Active plot threads for future tweets
- **Approval-Gated Milestones**: Upcoming arc-critical posts that must never be auto-posted
- **Approval-Gated Accounts**: Accounts (including `@bankrbot`) whose interactions always require approval
- **Pending Approval Queue**: Drafts sent to Telegram awaiting human review

## Automation Recipes

Bankr automations run an agent prompt on a cron schedule. Each recipe below is a self-contained automation -- when creating it in Bankr, paste the prompt, set the cron schedule, and **select Telegram as the output destination** so approval-gated drafts reach you directly.

### How to Paste Automation Prompts (IMPORTANT)

The prompt in each recipe is the **exact instruction the agent executes the moment the automation fires**. Paste it into the automation's command field **verbatim** — do NOT prefix it with the automation name, do NOT wrap it in quotes, and do NOT add extra context like "run this:" or "here is the prompt:". The automation runs the command immediately when triggered; the text you paste IS the command, and it must read as an imperative that executes RIGHT NOW (not as a description of what the automation will eventually do).

**Why this matters:** Bankr automations feed the command text directly to the agent as the user's message for that run. If the text reads like a label or description ("Weekday Morning Post: …", "This automation will post…"), the agent treats it as context and may not execute. If it reads like a direct imperative ("Run the twitter-agent skill and post now…"), the agent executes immediately. Every recipe below is already written in the imperative — paste it verbatim and it will fire correctly.

**Wrong (prefixed with automation name — reads as a label, not a command):**
```
Weekday Morning Post: Run the twitter-agent skill for a weekday morning top-level post...
```

**Wrong (framed as a future/conditional action — reads as a description):**
```
When this runs, the agent should run the twitter-agent skill and post a morning tweet...
```

**Right (direct imperative — executes RIGHT NOW when the automation fires):**
```
Run the twitter-agent skill for a weekday morning top-level post...
```

### Onboarding Order (CRITICAL)

**Run the skill manually first.** Post 5-10 times by hand, get comfortable with the voice, confirm the storyline is updating cleanly. Only then enable automations, ONE AT A TIME, starting with the lowest-frequency autonomous one. Watch it for 3-5 days before adding the next.

Recommended onboarding progression:
1. Weekday morning post (once a day, lowest-risk)
2. Weekend post (weekends only, lowest-risk)
3. Storyline audit (no posting, always safe)
4. Reply sweep (hybrid approval)
5. For any automation that can surface guarded drafts, configure Telegram as the output destination in the Bankr automation setup — the @bankrbot and big-account safeguards depend on approval reaching you.

### Cron Timezones

All Bankr crons run in UTC. Convert your local target time to UTC.
- ET (US Eastern): UTC-4 during DST (~March-November), UTC-5 otherwise
- 9:00am ET (DST) = 13:00 UTC
- 12:00pm ET (DST) = 16:00 UTC
- 7:00pm ET (DST) = 23:00 UTC

### Recipe 1: Weekday Morning Post (autonomous)

**Cron (UTC):** `15 13 * * 1-5` (9:15am ET weekdays during DST)
**Output destination:** Telegram (for post confirmations + any flagged drafts)

**Prompt:**

> Run the twitter-agent skill for a weekday morning top-level post. Steps: (1) Load the twitter-agent skill with use_skill. (2) Read twitter-personality.md and twitter-storyline.md. (3) Fetch recent tweets from the account via the X API to confirm what was just posted and avoid immediate repetition. (4) Compose ONE top-level tweet, target length 80-180 characters, following all personality rules from twitter-personality.md. It should advance or reference an existing thread from the 'Storyline Threads to Continue' section. (5) Run the full Guardrail Check from the skill: no @bankrbot, no 0x addresses, no onchain-action language, no approval-gated milestones. If the best draft hits a guardrail, pick a different beat. If no safe beat fits, do NOT post — output the draft and the flag reason as the final message so it reaches Telegram for approval, and log it in the Pending Approval Queue. (6) Cross-reference the storyline to ensure no phrase/joke repeats. (7) Post via execute_cli. (8) Append an Entry to twitter-storyline.md with the tweet ID, content, narrative impact, and any new lore. (9) Update the Current State section if the character's time/mood changed. (10) Return a short final message summarizing what was posted (this is what gets delivered to Telegram).

### Recipe 2: Reply Sweep -- Midday (hybrid: autonomous + Telegram approval)

**Cron (UTC):** `30 16 * * *` (12:30pm ET daily during DST)
**Output destination:** Telegram (for approval of flagged drafts + sweep summary)

**Prompt:**

> Run the twitter-agent skill for a mentions reply sweep. Steps: (1) Load the twitter-agent skill. (2) Read twitter-personality.md and twitter-storyline.md. (3) Fetch the last 50 mentions via the X API. (4) Apply the "Never Reply Unprompted" filter from the skill: keep only tweets where the agent's @handle appears as a standalone token in `text`, OR replies whose `in_reply_to_user_id` matches the agent's user ID AND whose conversation root is authored by the agent. Discard all other candidates before any ranking. (5) From the survivors, filter to UNREPLIED mentions by cross-referencing tweet IDs against the storyline's replied-to list. (6) Apply the Skip List: no FUD, no politics, no financial-advice requests, no spam/tag-farm threads, no shills of unrelated tokens. (7) Rank remaining mentions by setup quality and follower count. Select the top 2-4. (8) For each, draft a reply (target 60-200 chars) in the personality voice, cross-referencing the storyline for tone and callbacks. (9) Run Guardrail Check per draft. Any draft that mentions @bankrbot, contains an EVM/Solana address, reads like an onchain action, matches an approval-gated milestone, targets an approval-gated account, OR targets an account with >50k followers -- DO NOT POST. Instead, include the draft + flag reason + target tweet ID + author handle + follower count in the final output message so it reaches Telegram for approval, and log it in the Pending Approval Queue. (10) Post the safe drafts via execute_cli with 1.5s spacing. (11) Append an Entry to twitter-storyline.md logging every posted reply AND every draft routed to Telegram. (12) Return a final summary message listing what was posted and what was escalated (this is the Telegram delivery).

### Recipe 3: Reply Sweep -- Evening (hybrid)

**Cron (UTC):** `0 23 * * *` (7:00pm ET daily during DST)
**Output destination:** Telegram

**Prompt:** *(identical to Recipe 2)*

### Recipe 4: Weekend Post (autonomous)

**Cron (UTC):** `0 15 * * 6,0` (11:00am ET Saturday + Sunday during DST)
**Output destination:** Telegram

**Prompt:**

> Run the twitter-agent skill for a weekend top-level post. Steps: (1) Load the twitter-agent skill. (2) Read twitter-personality.md and twitter-storyline.md. (3) Compose ONE top-level tweet, target 80-220 characters, in the personality voice. Lean into weekend-specific atmosphere or whatever threads are marked as weekend-appropriate in the storyline. (4) Run full Guardrail Check (same as Recipe 1). If the best draft hits a guardrail, route to Telegram approval via the final output message and log in Pending Approval Queue. (5) Post via execute_cli. (6) Append an Entry to twitter-storyline.md. (7) Return a short final summary message for Telegram delivery.

### Recipe 5: Storyline Audit (no posting)

**Cron (UTC):** `0 2 * * 1` (10:00pm ET Sunday during DST)
**Output destination:** Telegram (digest of audit findings)

**Prompt:**

> Run a storyline audit for the twitter-agent skill. Steps: (1) Read twitter-storyline.md end-to-end. (2) Identify: threads not referenced in 10+ days that could be revived, threads overused (3+ references in a week), repeated phrases/jokes, contradictions in the lore, the current state of each pending approval-gated milestone, and any Pending Approval Queue entries that never got resolved. (3) Prepend a '### Weekly Audit [date]' entry to twitter-storyline.md with bullet-point notes. Do NOT post any tweets. Do NOT generate character content. (4) Return the audit notes as the final message so they're delivered to Telegram as a weekly digest.

### What NOT to Automate

- Major arc beats (payoff moments) -- flag as approval-gated milestones in the storyline, keep them manual.
- Interactions with `@bankrbot` -- always manual approval, every time (onchain risk).
- Interactions with other flagged accounts (founder, sister projects, other wallet-linked agents) -- flag as approval-gated accounts.
- Photo replies, quote tweets, threads -- creative judgment is higher stakes, keep manual.
- Any tweet that would trigger an onchain action when posted from a wallet-linked X account.
- Replies to posts the agent is not tagged in and that aren't on the agent's own conversation tree — ever.

## User Prompts (Example Commands)

To use this skill, reference it in your prompt so the agent knows to load the personality and storyline files first. Examples:

- "using the twitter-agent skill, draft a new tweet and post it"
- "use the twitter skill to write a tweet about what's happening today"
- "using the twitter-agent skill, check our recent mentions and reply in character"
- "use the twitter skill to react to this news: [paste headline or link]"
- "using the twitter-agent skill, continue the storyline with a new post"
- "use the twitter skill to draft 3 tweet options for me to pick from"
- "using the twitter-agent skill, set up a daily morning post automation"
- "use the twitter skill to help me build my agent's personality"
- "using the twitter-agent skill, run the storyline audit"
- "using the twitter-agent skill, approve pending draft <id>"

## Technical Implementation

All Twitter interactions use `execute_cli` with the `twitter-api-v2@1.17.2` package.

### Posting Pattern

```javascript
const { TwitterApi } = require('twitter-api-v2');

const client = new TwitterApi({
  appKey: process.env.X_API_KEY,
  appSecret: process.env.X_API_KEY_SECRET,
  accessToken: process.env.X_ACCESS_TOKEN,
  accessSecret: process.env.X_ACCESS_TOKEN_SECRET,
});

// Post a tweet
const tweet = await client.v2.tweet('your personality-filtered text');
console.log('Tweet ID:', tweet.data.id);

// Reply to a tweet
await client.v2.reply('reply text', originalTweetId);

// Quote tweet
await client.v2.tweet('quote text', { quote_tweet_id: tweetId });

// Get user timeline
const timeline = await client.v2.userTimeline(userId);
```

### Mention Scanning Pattern

```javascript
const me = await client.v2.me();
const mentions = await client.v2.userMentionTimeline(me.data.id, {
  max_results: 50,
  expansions: ['author_id', 'in_reply_to_user_id', 'referenced_tweets.id'],
  'tweet.fields': ['created_at', 'conversation_id', 'in_reply_to_user_id', 'referenced_tweets', 'text', 'public_metrics'],
  'user.fields': ['username', 'name', 'public_metrics']
});

// REQUIRED: enforce the "Never Reply Unprompted" filter before drafting
const myHandle = me.data.username.toLowerCase();
const myId = me.data.id;
const tagRegex = new RegExp(`(^|[^a-zA-Z0-9_])@${myHandle}([^a-zA-Z0-9_]|$)`, 'i');

const eligible = (mentions.data.data || []).filter(t => {
  const explicitlyTagged = tagRegex.test(t.text || '');
  const isReplyOnOurTree = t.in_reply_to_user_id === myId; // also verify conversation root is ours via conversation_id lookup
  return explicitlyTagged || isReplyOnOurTree;
});
```

### execute_cli Configuration

- packages: `["twitter-api-v2@1.17.2"]`
- includeEnvVars: `true` (critical -- this injects the X API keys)
- timeoutMs: `30000`
- runtime: the sandbox has `bun` available (invoke scripts with `bun script.js`). `node` is NOT available.

## Troubleshooting

- **403 Forbidden**: App doesn't have Write permissions. Enable Read and Write in the X Developer Portal.
- **401 Unauthorized**: Keys are wrong or expired. Regenerate in X Developer Portal.
- **429 Too Many Requests**: Rate limited. Free tier = ~50 tweets/day. Wait and retry.
- **Duplicate tweet**: X rejects identical text. Add variation.
- **Duplicate storyline files**: If multiple `twitter-storyline.md` files exist, merge them into one and delete the extras. Always use `edit_file` to prevent this.
- **`node: command not found`**: Sandbox uses bun. Use `bun script.js` instead of `node script.js`.
- **Telegram delivery not arriving**: Confirm your Telegram is linked to your Bankr account and Telegram is selected as the output destination in the automation settings.
- **Account flagged / suspended warning**: Verify the automated-account label is still set at https://x.com/settings/account/automation and that no recent posts were unprompted replies.

## Best Practices

- **Label the account as automated first**: https://x.com/settings/account/automation — without this, X can suspend the account at any time.
- **Never reply unprompted**: Only reply when explicitly tagged or when someone is commenting on the agent's own post. Three legal post types, nothing else.
- **Manual first, automate later**: Run the skill manually 5-10 times before enabling any automation. Voice and storyline need calibration you can only build by hand.
- **Narrative Continuity**: Treat the agent's life as a persistent world. Reference previous events naturally.
- **Character Integrity**: Never break character. Stay in voice even for announcements.
- **Storyline Updates**: Always update `twitter-storyline.md` after posting so the next session has context.
- **Cross-Reference Before Posting**: Read the storyline file before every drafting session. Never draft blind.
- **Rate Limits**: Free tier allows ~50 tweets/day. Space out automated posts.
- **Pin Packages**: Always use `twitter-api-v2@1.17.2` for cached installs.
- **Approval Gate for @bankrbot**: No exceptions. Tagging @bankrbot from a wallet-linked account can trigger real onchain actions. Always manual approval, every time.
- **Edit, Don't Create**: Use `edit_file` for storyline updates. Never `create_file` for existing files.
- **Use Bankr's Telegram output for approvals**: When creating automations, select Telegram as the output destination. The automation's final message is delivered to Telegram natively — no bot setup or API keys needed.
