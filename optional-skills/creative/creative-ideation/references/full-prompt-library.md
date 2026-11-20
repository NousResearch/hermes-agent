# Constraint Library

Constraint-dispatch library — voice and approach inspired by [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html). Adapted and expanded.

Constraint plus direction is creativity. Pick a constraint, generate 3 ideas that satisfy it, ship one.

## How to use

The library is split by **domain affinity**:

- **General** — works for any domain. Default for SPECIFICITY=NONE.
- **Software / artifact** — when DOMAIN=ARTIFACT.
- **Physical / object** — when DOMAIN=OBJECT.
- **Social / collective** — when work involves other people.
- **Lists** — domain-agnostic, more whimsical.

When in doubt: pick one from General. When the user has stated a domain, pick from that domain's section. Pick by random, by mood match, or by what's nearest the user's wording. Don't enumerate all of them.

Every prompt is interpreted as broadly as possible. "Does this include X?" → yes. The constraints provide direction and mild constraint; both are needed.

---

## General — any domain (default)

**Start at the punchline.**
Think of something that would be a funny sentence. Work backwards to make it real. *"I taught my thermostat to gaslight me"* → now build it.

**High concept, low effort.**
A deep idea, lazily executed. The concept should be brilliant. The implementation should take an afternoon. If it takes longer, you're overthinking it.

**Take two.**
Remember an old project of yours. Do it again from scratch. No looking at the original. See what changed about how you think.

**Blatantly copy something.**
Pick something you admire — a tool, an artwork, an interface. Recreate it from scratch. The learning is in the gap between your version and theirs.

**Translate.**
Take something meant for one audience and make it understandable by another. A research paper as a children's book. An API as a board game. A song as an architecture diagram.

**Make a self-portrait.**
Be yourself? Be fake? Be real? In code, in data, in sound, in a directory structure, on paper, in clay.

**Make a mirror.**
Something that reflects the viewer back at themselves. A website that shows your browsing history. A CLI that prints your git sins. A garment that changes color based on the wearer's heart rate.

**Make a pun.**
The stupider the better. Physical, digital, linguistic, visual. The project IS the joke.

**Hostile UI.**
Make something intentionally painful to use. A password field that requires 47 conditions. A form where every label lies. A door that judges you. The cruelty is the design.

**The useless tree.**
Make something useless. Deliberately, completely, beautifully useless. No utility. No purpose. No point. That's the point.

**One million of something.**
One million is both a lot and not that much. One million pixels is a 1MB photo. One million API calls is a Tuesday. One million of anything becomes interesting at scale.

**Make something that dies.**
A website that loses a feature every day. A chatbot that forgets. A countdown to nothing. A garment that wears out as it's worn. An exercise in rot, killing, or letting go.

**Doors, walls, borders, barriers, boundaries.**
Things that intermediate two places: opening, closing, permeating, excluding, combining.

**Borges week.**
Something inspired by the Argentine. The library of Babel. The map that is the territory. Two writers separated by 400 years writing the same book.

**An idea that comes from a book.**
Read something — anything, deeply, even a footnote. Make something inspired by it.

**Go to a museum.**
Project ensues.

**Office Space printer scene.**
Capture the same energy. Channel the catharsis of destroying the thing that frustrates you.

**NPC loot.**
What do you drop when you die? What do you take on your journey? Build the item.

**Mythological objects and entities.**
Pandora's box, the ocarina of time, the palantir, the sword in the stone, the seal of Solomon. Build the artifact.

**The more things change, the more they stay the same.**
Reflect on time, difference, and similarity. Same neighborhood different decade. Same recipe different cook.

---

## Software / artifact (DOMAIN=ARTIFACT)

**Solve your own itch.**
Build the tool you wished existed this week. Under 50 lines. Ship it today.

**Automate the annoying thing.**
What's the most tedious part of your workflow? Script it away. Two hours to fix a problem that costs you five minutes a day.

**The CLI tool that should exist.**
Think of a command you've wished you could type. `git undo-that-thing-i-just-did`. `docker why-is-this-broken`. `npm explain-yourself`. Now build it.

**Nothing new except glue.**
Make something entirely from existing APIs, libraries, and datasets. The only original contribution is how you connect them.

**Frankenstein week.**
Take something that does X and make it do Y. A git repo that plays music. A Dockerfile that generates poetry. A cron job that sends compliments.

**Subtract.**
How much can you remove from a codebase before it breaks? Strip a tool to its minimum viable function. Delete until only the essence remains.

**Something for your desktop.**
You spend a lot of time there. Spruce it up. A custom clock, a pet that lives in your terminal, a wallpaper that changes based on your git activity.

**One screen, two screen, old screen, new screen.**
Take something you associate with one screen and put it on a very different one. DOOM on a smart fridge. A spreadsheet on a watch. A terminal in a painting.

**Code as koan, koan as code.**
What is the sound of one hand clapping? A program that answers a question it wasn't asked. A function that returns before it's called.

**Artificial stupidity.**
Make fun of AI by showcasing its faults. Mistrain it. Lie to it. Build the opposite of what AI is supposed to be good at.

**"I use technology in order to hate it properly."**
Make something inspired by the tension between loving and hating your tools.

**I mean, I GUESS you could store something that way.**
The project works when you can save and open something. Store data in DNS caches. Encode a novel in emoji. Write a file system on top of something that isn't a file system.

**I mean, I GUESS those could be pixels.**
The project works when you can display an image. Render anything visual in a medium that wasn't meant for rendering.

**Text is the universal interface.**
Build something where text is the only interface. No buttons, no graphics, just words in and words out. Text can go in and out of almost anything.

---

## Physical / object (DOMAIN=OBJECT)

**Do a lot of math.**
Generative geometry, shader golf, mathematical art, computational origami. Time to re-learn what an arcsin is.

**Lights!**
LED throwies, light installations, illuminated anything. Make something that glows.

---

## Social / collective

**Create a means of distribution.**
The project works when you can use what you made to give something to somebody else.

**Make a way to communicate.**
The project works when you can hold a conversation with someone else using what you created. Not chat — something weirder.

**Write a love letter.**
To a person, a programming language, a game, a place, a tool. On paper, in code, in music, in light. Mail it.

**Mail chess / asynchronous games.**
Something turn-based played with no time limit. No requirement to be there at the same time. The game happens in the gaps.

**Twitch plays X.**
A group of people share control over something. Collective input, emergent behavior.

---

## Lists (any domain, slightly more whimsical)

**Lists!**
Itemizations, taxonomies, exhaustive recountings, iterations. This one. A list of list of lists.

**Did you mean *recursion*?**
Did you mean recursion?

**Animals.**
Lions, and tigers, and bears. Crab logic gates. Fish plays the stock market.

**Cats.**
Where would the internet be without them.

---

## Attribution

Constraint approach inspired by [wttdotm.com/prompts.html](https://wttdotm.com/prompts.html). Original v1 of this library was substantially adapted from there. This expanded version groups constraints by domain affinity for use with the routing logic in `SKILL.md`.
