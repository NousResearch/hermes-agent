---
name: html5-canvas-game-dev
description: >
  Build single-file HTML5 canvas games. Covers survivor shooters,
  artillery/Worms games, shop and perk systems, Web Audio sound,
  mathematical progression, and roguelite design. Patterns from
  real community-built games using Hermes Agent.
version: 2.0.0
license: MIT
tags: [gamedev, html5, canvas, javascript, game-design, web-audio, worms, artillery]
metadata:
  hermes:
    category: gaming
    triggers:
      - html game
      - canvas game
      - browser game
      - shoot em up
      - brotato
      - worms game
      - artillery game
      - turn based game
      - game balance
      - web audio sounds
---

# HTML5 Canvas Game Development

Patterns for building single-file HTML5 canvas games. No build tools,
no dependencies — everything in one .html file.

For detailed code snippets, see [references/code-patterns.md](references/code-patterns.md).

---

## 1. Architecture

Single HTML file with: `<style>` (overlays), `<canvas>`, `<script>` (everything).

Game state machine (survivor):
```
breedSelect → playing → shop → playing → ... → boss → victory → newRun
                ↓                                                  ↓
             gameOver → restart → breedSelect              breedSelect
```

Game state machine (artillery/turn-based):
```
lobby → teamSelect → myTurn ↔ theirTurn → gameOver
```

Key principle: Use delta time (`dt`) for ALL movement and timers.
`entity.x += speed * dt` — never assume a fixed frame rate.

---

## 2. Projectile Systems

Every projectile is an object in a `projectiles[]` array.

Base properties: x, y, vx, vy, damage, range, distance, radius,
pierce, bounce, knockback, slow, weaponType, explosive, expSize.

Special behaviors (checked each frame):

| Mechanic | Summary |
|----------|---------|
| Pierce | `p.pierce--` on hit; remove at `< 0` |
| Bounce | Flip velocity on wall; `p.bounce--` |
| Homing | Steer toward nearest enemy |
| Gravity | `p.vy += gravity * dt` (arcs, cannons) |
| Boomerang | Decelerate, reverse, catch on return |
| Ricochet | Redirect toward next enemy after hit |
| Fire Zone | Create persistent AoE on impact |
| Spin-up | Ramp fire rate from 30%→100% over time |
| KB Cap | Only strongest knockback per enemy per frame |
| Wind | `p.vx += wind * strength * dt` (artillery) |
| Charge | Hold to charge power, release to fire |

See code-patterns.md for full implementations.

---

## 3. Enemy AI Patterns

Common types:
- **Melee**: Walk toward player, attack on timer at melee range
- **Ranged**: Approach to preferred distance, shoot, retreat if too close
- **Rush**: Wobble movement with sin() offset, speed burst when close
- **Splitter**: On death spawn N smaller enemies (use pendingSpawns array)
- **Tank**: Slow, high HP, knockback resistant
- **Explosive**: Approach → windup telegraph → dodgeable dash

Dodge system: Check at EVERY damage source. Show "DODGE!" text feedback.
Players won't know the stat works without visible confirmation.

Collision fix: Push enemies out when overlapping player body.

---

## 4. Mathematical Progression

Five systems for organic, non-linear game feel:

| System | Purpose | Formula |
|--------|---------|---------|
| Golden Ratio (φ) | Economy scaling | `cost = base * φ^count` |
| Sinusoidal Waves | Spawn rhythm | `\|sin(N*π*t)\|^3` = N surge humps |
| Logistic Map | Chaos at high diff | `x = r*x*(1-x)`, r > 3.57 = chaos |
| Prime Factorization | Mutation combos | Each prime = mutation type |
| Harmonic Series | Diminishing returns | `value = base / (1 + count)` |

Golden ratio drives costs UP, harmonic drives value DOWN. Together
they create a natural soft cap — buying more is always possible but
increasingly inefficient.

Wave spawning: Use TIME-based progress (waveTimer/waveDuration), not
enemy count. This makes rhythm predictable regardless of kill speed.

---

## 5. Sound Design (Web Audio API)

No audio files needed. Oscillator + gain envelope = full sound library.

Rules of thumb:
- Positive events: ASCENDING frequency (coins, kills, victory)
- Negative events: DESCENDING frequency (hurt, death)
- Impact events: Start LOUD, decay fast
- Frequent sounds: 0.03-0.1s. Rare sounds: 0.3-1s
- Throttle rapid sounds to prevent audio overload

Pitch-scaling pickups: Rapid coin collection raises pitch (+30hz per
pickup, cap at +400hz, reset after 0.4s gap). Sells the "vacuum" feel.

Catch sounds: When a boomerang returns, play a short ascending click
(800→1100hz, 0.07s). Validates the mechanic — players feel the catch.

Easter eggs: Buffer keystrokes during specific states. Trigger on
secret word match. Show brief visual confirmation.

---

## 6. Shop / Perk Systems

Shop flow: Wave ends → open shop → buy/sell/refresh/lock → next wave.

Perk gating: Higher ranks unlock at later waves.
Duplicate protection: `ownedPerkKeys` Set.
Pity system: Force high-rank after N low shops.

### Perk Taxonomy (complete survivor game needs ALL of these)

| Category | Examples |
|----------|---------|
| Offense | damage, crit chance, crit damage, fire rate |
| Defense | max HP, regen, dodge, armor (flat reduction) |
| Sustain | lifesteal (heal % of damage dealt) |
| Shield | triggered shield at low HP with cooldown |
| Thorns | reflect damage on melee hit |
| Utility | speed, luck, magnet, projectile bounce |
| Economy | passive coin gen, shop discount |
| Risk/Reward | trade HP for coins, conditional damage bonus |
| Legendary | build-defining uniques (1 per run) |

### Key Design Patterns

**Immediate-effect perks**: Not all perks are passive. Some trigger on
purchase (e.g., lose 10% max HP, gain 50 coins NOW). Show both the
gain and loss as floating damage text — player must see the tradeoff.

**Conditional damage (Leverage)**: +60% damage above 80% HP, -20% below
40%. Creates synergy trees: leverage + lifesteal = stay topped off.

**Repeatable perks**: When all uniques exhausted, offer stackable perks
with golden ratio costs and harmonic diminishing returns.

### Perk Display: Sticker Collage

Render perks as colored badge stickers. Under 8 perks: clean stack.
8+ perks: "collage mode" with deterministic random rotations and
overlapping like a poster board. Higher rank = slightly bigger + higher
z-index. Rank 5 gets a glow animation.

---

## 7. Roguelite Progression

**Victory carryover** (beating the boss):
- Save 10% of coins (rounded up)
- Carry weakest weapon (reset to level 1)
- Carry all perks
- Increment difficulty

**Death** = lose everything, reset to difficulty 1.

**Weapon graduation**: After difficulty N, remove starter weapon from
shop. Player has outgrown it. Keeps shop interesting.

---

## 8. Worms / Artillery Games (Turn-Based)

Turn system: 40-second timer + move budget (800 units per turn).
End turn on timer expiry, manual button, or after projectile lands.

Core mechanics:
- **Aim**: Arrow keys rotate aim angle (clamp to upward arc)
- **Charge-to-fire**: Hold Space to charge power (0-100%), release to fire
- **Wind**: Random per turn, affects horizontal velocity
- **Trajectory**: Gravity pulls down, wind pushes sideways
- **Destructible terrain**: Heightmap array, lower values in blast radius
- **Limited ammo**: Infinite fallback weapon + limited special weapons

Multiplayer: WebSocket or Supabase realtime for turn sync. Support
guest login + social auth. Private game links via UUID room codes.
Global leaderboard with win/loss tracking.

---

## 9. Balance

DPS = damage × fireRate. Account for pierce, pellets, splash.
Per upgrade level: damage × 1.25, fireRate × 1.12.

Red flags:
- Any weapon > 2x DPS of same tier = nerf it
- Zero downsides + top DPS = broken
- "Fun but weak" beats "boring but strong"

Difficulty scaling (use multiple levers):
- HP: linear (main lever)
- Speed: logarithmic `speed * (1 + log(diff) * 0.08)`
- Damage: logarithmic `damage * (1 + log(diff) * 0.12)`

---

## 10. Community Showcase

Games built with Hermes Agent and these patterns:

**HODL OR DIE (BrotherDoge)** — @123mikeyd
Brotato-style survival shooter. Dogecoin themed. Wave-based with
shop system, 11 weapons, 40+ perks, boss battles, roguelite
carryover, sticker perk collage, degen easter egg.
https://github.com/123mikeyd/Brother_Doge_VideoGame

**WAR_V3_FINALE** — @javierblez
Worms-style 2D artillery. Built in 2.5 hours with Hermes Agent.
Turn-based with wind, destructible terrain, 4 weapons, multiplayer
matchmaking, leaderboards, and private game links.
https://warv3finale.com
