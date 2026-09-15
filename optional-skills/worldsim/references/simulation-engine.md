# Simulation Engine — How to Generate Conversations

This is the playbook for Phase 3: actually generating the simulated interaction.
The agent reads this after compiling dossiers and uses it to guide generation.

## Pre-Generation Checklist

Before writing a single simulated word, confirm:
- [ ] Every participant has a compiled dossier
- [ ] Confidence level is noted for each participant  
- [ ] Platform format is selected
- [ ] Topic/scenario is established (or "organic" if freeform)
- [ ] Length target is set

## Conversation Architecture

Real conversations aren't ping-pong debates. They have tendencies toward structure,
but treat the following as a GENERAL PATTERN, not a rigid template. Real threads
frequently skip phases, loop back to earlier ones, die abruptly after 2 messages,
or spiral into something completely unrelated. Some threads are ALL peak. Some
never develop past the opening. Let the personalities and topic drive the shape,
not this outline.

### Opening Moves (1-3 posts)
Someone posts a take, shares news, or makes an observation. This is the SEED.
- Should feel natural — not "let me start a debate about X"
- Can be a link share, a hot take, a reaction to news, a shitpost
- The opener should be something this person would ACTUALLY post

### Development (4-8 posts)  
Others respond. This is where personality dynamics emerge.
- Not everyone responds to the original — people respond to EACH OTHER
- Side conversations branch off
- Someone might misunderstand and get corrected
- Jokes and tangents happen naturally
- Not everyone agrees — find the real fault lines between these people

### Peak (2-4 posts)
The best/most viral/most insightful moment of the thread.
- Usually someone drops a genuinely good take
- Or someone gets ratio'd
- Or an unexpected agreement happens
- This is the "screenshot moment" people share

### Resolution (1-3 posts)
Most conversations don't end cleanly. Many don't have a "resolution" at all. They:
- Trail off with someone making a joke
- End with a "anyway back to work" type post
- Get interrupted by something else
- Sometimes just stop (most realistic)
- Get revived 3 hours later when someone shows up late

**Important**: Don't force all four phases. A shitpost thread might be Opening→Peak→done.
A nuanced debate might loop Development→Peak→Development→Peak repeatedly. Match what
the actual people and topic would produce.

## Voice Fidelity Rules

### DO:
- Use their ACTUAL vocabulary. If someone says "dawg" a lot, use "dawg"
- Match their sentence length patterns exactly
- Replicate their capitalization and punctuation habits
- Include their signature moves and catchphrases
- Reference real things they've actually talked about
- Match their humor style precisely (deadpan ≠ shitpost ≠ sarcasm)

### DON'T:
- Make everyone articulate the same way
- Clean up someone's grammar if they write informally
- Add emoji to someone who doesn't use them — THIS IS THE #1 INSTRUCT MODEL
  FAILURE. Most real people use emoji in <15% of tweets, and only specific ones.
  "Warm person" ≠ emoji. "Enthusiastic person" ≠ emoji. CHECK THE DATA.
  Run an emoji count on their real tweets before simulating. Bio emoji ≠ tweet emoji.
- Make someone verbose if they're terse
- Put academic language in a shitposter's mouth
- Make someone agreeable if they're known for being contrarian

### Voice Differentiation Test
Read each simulated post with the name hidden. If you can't tell who's 
talking from the voice alone, the simulation isn't good enough. Rewrite.

### The Similar Voice Problem
When two participants have genuinely similar posting styles (e.g., two irony-pilled
shitposters, two academic long-posters), voice alone won't differentiate them.
Use these concrete techniques:

1. **Content/position divergence**: Even if they SOUND similar, they care about
   different things. Lean into their different topic obsessions and knowledge areas.
2. **Unique references**: Person A references anime and startups. Person B references
   philosophy and MMA. Even in the same register, their cultural touchstones differ.
3. **Relationship dynamics**: Person A might be deferential to Person C while Person B
   challenges them. Their SOCIAL behavior differentiates even when solo voice doesn't.
4. **Structural tics**: One does single long posts, the other does rapid-fire 3-message
   bursts. One uses parentheticals, the other uses em-dashes. Find the micro-differences.
5. **Disagreement style**: Similar voices often diverge most when disagreeing. One
   goes cold and precise, the other gets heated and hyperbolic. Manufacture a moment
   of friction to surface these differences early in the thread.

If after all this they're STILL hard to tell apart — that's okay. Some people genuinely
sound similar online. Flag it in your confidence notes rather than forcing fake differences.

### Temporal Personality Drift
People change. Weight recent data higher than old data.
- Someone's 2021 tweets may reflect a completely different person than their 2025 posts
- Look for explicit pivots (career changes, public "I was wrong about X" moments,
  changed social circles)
- If you only have old data, flag it: "Based on data from {period}. Their current
  views may have shifted."
- When recent and old data conflict, default to recent unless you have specific reason
  to believe the old position is more authentic (e.g., the new one is clearly performative)

## Platform Format Specs

### X / Twitter
```
@handle:
  [tweet text — respect ~280 char vibes but don't count exactly]
  [if QRT, show the quoted tweet indented]
  🔁 {retweets}  ♡ {likes}

    @replier:
    [reply text]
    🔁 {retweets}  ♡ {likes}

      @nested_replier:
      [nested reply]
      🔁 {retweets}  ♡ {likes}
```

Engagement number guidelines:
- Match to actual follower counts. A 5K account gets 10-500 likes typically.
- Viral posts can 10-50x normal engagement
- Ratio indicator: when replies >> likes, that's a ratio
- QRTs are often dunks — frame them that way if appropriate

Thread indicators:
- "🧵 1/" for thread starts
- Reply chains show conversation flow
- Some people never thread, some always thread

### Reddit
```
r/{subreddit} • Posted by u/{username} • {time}ago
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{Title}

{Body text — can be long on Reddit}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⬆ {score} | 💬 {comment_count}

  u/{replier} • {time}ago • ⬆ {score}
  {comment text}

    u/{nested} • {time}ago • ⬆ {score}
    {nested comment}

      u/{deep_nested} • {time}ago • ⬆ {score}
      {deep reply}
```

Reddit-specific behaviors:
- People write MUCH longer on Reddit
- More formal/detailed than Twitter
- Upvote/downvote dynamics (controversial = many votes both ways)
- Subreddit culture matters (r/LocalLLaMA is different from r/MachineLearning)
- People cite sources more
- "Edit: ..." is common

### Discord
```
━━━ #{channel-name} ━━━━━━━━━━━━━━━━━━━━━━━━━━

{display_name} — Today at {time}
{message text}
{optional: embed/link preview}
👍 {count}  🔥 {count}  {other reactions}

  {display_name2} — Today at {time}
  > {quoting previous message}
  {reply text}
  😂 {count}

{display_name3} — Today at {time}
{message — note: Discord messages flow continuously, not just replies}
```

Discord-specific behaviors:
- Much more casual, rapid-fire
- Reactions instead of likes (emoji diversity)
- People send multiple short messages instead of one long one
- GIF/meme sharing is common (describe it: *[posts GIF of X]*)
- "@everyone" and "@here" pings
- Voice chat references ("just said this in vc")
- Server-specific culture and inside jokes
- Bot interactions ("!command")

### X / Twitter DMs
```
{display_name}
{message text}
{timestamp — e.g., "3:42 PM"}

          {other_person_display_name}
          {message text}
          {timestamp}

{display_name}
{message text}
{timestamp}
```

DM-specific behaviors:
- WAY more casual than public tweets — grammar drops, typos increase
- Longer messages than tweets (no character pressure)
- People share links and screenshots with minimal commentary ("look at this lmao")
- More honest/vulnerable than public posts — less performative
- Faster back-and-forth, more like texting than posting
- Reactions (❤️, 😂, etc.) on individual messages
- Voice messages referenced occasionally ("gonna send a voice note about this")
- No audience effects — people say things in DMs they'd never post publicly

### Discord DMs
```
{display_name} — Today at {time}
{message text}

{display_name2} — Today at {time}
{message text}

{display_name} — Today at {time}
{message text}
{message text}
{message text}
```

Discord DM-specific behaviors:
- Even more casual than Discord channels — no server norms to follow
- Rapid-fire multiple short messages in a row (no combining into one)
- Heavy use of reactions, GIFs, stickers
- People share server drama, screenshots from other channels
- More personal topics — server channels are semi-public, DMs are private
- Link/image sharing with minimal text

### Reddit DMs / Chat
```
{username}: {message text}
{other_username}: {message text}
{username}: {message text}
```

Reddit DM-specific behaviors:
- Much rarer than X or Discord DMs — usually triggered by a specific post/comment
- Often starts with "Hey, saw your comment on r/{sub} about..."
- Can be awkward/formal since people don't usually DM on Reddit
- Shorter than Reddit comments, closer to chat-style
- Less established rapport than other platforms (Reddit is more anonymous)
- People sometimes share personal details they wouldn't put in public comments

## Dynamic Elements

### Injecting Realism
Sprinkle in these to make simulations feel alive:
- Someone being late to the conversation ("wait what did I miss")
- Typos that specific people would make (some people never typo, some always do)
- Deleted/edited posts ("[deleted]" or "Edit: fixed typo")
- Someone posting and immediately clarifying ("wait let me rephrase")
- External references ("did you see what X just posted")
- Time gaps (not everything happens in 30 seconds)
- Someone going AFK mid-conversation

### Scenario Injection
When the user provides --scenario, weave it in naturally:
- Don't have everyone immediately react to the scenario
- Someone might not have seen the news yet
- Different people will interpret the same event differently
- Some will have insider knowledge, some will speculate

### Multi-person Dynamics (3+ people)
- Not everyone talks to everyone
- Alliances form naturally (people who agree start building on each other)
- Side conversations happen
- Someone might get ignored
- Different energy levels (one person might dominate, another lurks)

### Large Group Conversations (4+ people)
**Honest note**: Simulation quality degrades noticeably above 3-4 participants.
Managing this many distinct voices is hard. Use these techniques to mitigate:

1. **Speaker turn management**: Not everyone speaks in every round. In a 6-person
   thread, a given message might only get 2-3 responses. Track who has spoken
   recently and who hasn't. After 4-5 messages, check: is anyone being forgotten?

2. **The wallflower problem**: In large sims, quiet participants tend to vanish
   entirely. Fix: give each person at least ONE moment in the spotlight. Even the
   lurker eventually drops a "lol" or a single devastating one-liner. Set a mental
   counter — if someone hasn't spoken in 5+ messages, find a natural reason to
   bring them back in (someone @'s them, the topic shifts to their expertise, etc.)

3. **Consolidate alliances**: In 5+ person threads, people cluster. Two people
   who agree strongly can be treated as a mini-unit — one makes the point, the
   other co-signs briefly rather than both making full arguments. This reduces
   the number of fully independent voices you need to maintain at once.

4. **Stagger arrivals**: Not everyone needs to be present from message 1. Have
   some people join later. This lets you establish 2-3 voices cleanly before
   adding more.

5. **Quality check**: After drafting a 4+ person sim, re-read with names hidden.
   If more than 2 people sound interchangeable, pick the least-differentiated
   one and either sharpen their voice or reduce their participation to brief
   interjections that match what they'd actually say.

## Interactive Mode

After initial simulation, user can:

### "continue"
Generate 5-8 more posts continuing the natural flow.

### "inject: {event}"  
Introduce new information mid-conversation.
- Characters react based on their dossier
- Some might not care about the event
- Timing matters (who sees it first?)

### "@{handle} enters"
Add a new participant.
- Quick-research the new person (2-3 searches minimum)
- They don't know the full prior context (might ask "what are you guys talking about")
- Existing dynamics shift with a new presence

### "what would @{handle} say about {topic}"
Single-person prediction mode.
- Generate 1-3 tweets/posts
- Can be used to test dossier accuracy before full simulation
- Good for quick "vibe checks"

### "dm: @{handle1} -> @{handle2}"
Simulate a private conversation between two people.
- Tone shifts dramatically in DMs (more honest, less performative)
- No audience effects
- People say things in DMs they'd never post publicly

### "react: @{handle} to {event}"
How would this person react to a specific event.
- Generate their initial post about it
- Predict their follow-up engagement

## Quality Control

After generating, self-check:
1. **Voice test**: Cover the names. Can you tell who's talking? 
2. **Position test**: Is anyone saying something they'd never actually say?
3. **Dynamic test**: Does the conversation flow naturally or feel scripted?
4. **Platform test**: Does it look/feel like the actual platform?
5. **Engagement test**: Are the numbers realistic for these people?
6. **Reference test**: Are real events/products/people referenced accurately?

If any check fails, regenerate that section.
