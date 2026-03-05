---
name: pokemon-player
description: Play Pokémon games autonomously via headless emulation. Starts a game server, reads structured game state from RAM, makes strategic decisions, and sends button inputs — all from the terminal.
tags: [gaming, pokemon, emulator, pyboy, gameplay, gameboy]
---
# Pokémon Player

Play Pokémon games via headless emulation using the `pokemon-agent` package.

## When to Use
- User says "play pokemon", "start pokemon", "pokemon game"
- User asks about Pokemon Red, Blue, Yellow, FireRed, etc.
- User wants to watch an AI play Pokemon
- User references a ROM file (.gb, .gbc, .gba)

## First-Time Setup

### 1. Install the package
```bash
pip install pokemon-agent[dashboard] pyboy
```

### 2. Get the ROM
Ask the user for their ROM file path. Do NOT attempt to download ROMs.

### 3. Start the game server
```bash
pokemon-agent serve --rom <ROM_PATH> --port 8765 &
```
Wait 3 seconds, then verify:
```bash
curl -s http://localhost:8765/health
```

## The Gameplay Loop

### Step 1: OBSERVE
```bash
curl -s http://localhost:8765/state
```

### Step 2: ORIENT
- Dialog active → advance text
- In battle → fight
- Party hurt → heal
- Near objective → navigate

### Step 3: DECIDE
Priority order:
1. If dialog active → a_until_dialog_end
2. If in battle → choose best move
3. If any Pokemon <20% HP → Pokémon Center
4. If near story objective → navigate to it
5. If underleveled → train in grass
6. Otherwise → explore

### Step 4: ACT
```bash
curl -s -X POST http://localhost:8765/action \
  -H "Content-Type: application/json" \
  -d '{"actions": ["walk_up", "walk_up", "press_a"]}'
```

Action reference:
- press_a — confirm, talk, select
- press_b — cancel, close menu
- press_start — open game menu
- walk_up/down/left/right — move one tile
- a_until_dialog_end — advance all dialog
- wait_60 — wait ~1 second

### Step 5: VERIFY
Check state_after in the response. If stuck 3+ turns:
1. Press B several times
2. Try different directions
3. Take screenshot and use vision_analyze
4. Load last save if truly stuck

### Step 6: RECORD
```
memory add: PKM:OBJECTIVE: Heading to Pewter City to challenge Brock
memory add: PKM:PROGRESS: Got Squirtle, Got Pokedex, → Pewter City
```

### Step 7: SAVE
Save every 20-30 turns and ALWAYS before gym battles:
```bash
curl -s -X POST http://localhost:8765/save \
  -H "Content-Type: application/json" \
  -d '{"name": "before_brock"}'
```

## Battle Strategy

### Decision Tree
1. Want to catch? → Weaken then throw Poké Ball
2. Wild you don't need? → RUN
3. Type advantage? → Use super-effective move
4. No advantage? → Use strongest STAB move
5. Low HP? → Switch or use Potion

### Type Chart
- Water beats Fire, Ground, Rock
- Fire beats Grass, Bug, Ice
- Grass beats Water, Ground, Rock
- Electric beats Water, Flying
- Ground beats Fire, Electric, Rock, Poison
- Psychic beats Fighting, Poison (dominant in Gen 1!)

### Gen 1 Quirks
- Special stat is both offense AND defense for special moves
- Psychic is overpowered (Ghost moves bugged)
- Critical hits based on Speed stat
- Wrap/Bind prevent opponent from acting

## Memory Conventions
| Prefix | Purpose | Example |
|--------|---------|---------|
| PKM:OBJECTIVE | Current goal | Defeat Brock in Pewter City |
| PKM:MAP | Navigation knowledge | Viridian Forest: go north |
| PKM:STRATEGY | Battle/team plans | Need Grass type before Misty |
| PKM:PROGRESS | Milestone tracker | ✓ Boulder Badge → Cascade Badge |
| PKM:STUCK | Stuck situations | Got stuck in Cerulean Cave |
| PKM:TEAM | Team notes | Squirtle is Water/Ice coverage |

## Progression Milestones
- ☐ Choose starter
- ☐ Deliver Oak's Parcel → receive Pokédex
- ☐ Boulder Badge — Brock (Rock) → use Water/Grass
- ☐ Cascade Badge — Misty (Water) → use Grass/Electric
- ☐ Thunder Badge — Lt. Surge (Electric) → use Ground
- ☐ Rainbow Badge — Erika (Grass) → use Fire/Ice/Flying
- ☐ Soul Badge — Koga (Poison) → use Ground/Psychic
- ☐ Marsh Badge — Sabrina (Psychic)
- ☐ Volcano Badge — Blaine (Fire) → use Water/Ground
- ☐ Earth Badge — Giovanni (Ground) → use Water/Grass/Ice
- ☐ Elite Four → Champion!

## Stopping Play
1. Save the game:
```bash
curl -s -X POST http://localhost:8765/save \
  -d '{"name": "session_end"}'
```
2. Update memory with progress
3. Tell user: "Game saved! Say 'play pokemon' to resume."
4. Kill the background server process

## Dashboard
If `pokemon-agent[dashboard]` is installed, open:
http://localhost:8765/dashboard

Live features: game screen, AI reasoning stream, team status, action log.

## Pitfalls
- NEVER download or provide ROM files — always ask the user
- Don't send more than 15 actions per /action call
- Always wait for dialog to clear before moving
- Save BEFORE gym battles
- Take screenshots sparingly — they cost vision tokens
- Verify server is running with /health before any commands
