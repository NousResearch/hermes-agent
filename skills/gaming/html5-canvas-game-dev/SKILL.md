---
name: html5-canvas-game-dev
description: >
  Build and iterate on single-file HTML5 canvas games. Covers architecture,
  projectile systems, enemy AI, mathematical progression, Web Audio synth
  sounds, shop/perk systems, visual juice, and balance methodology.
  Built from real experience developing a Brotato-style shoot-em-up.
tags: [gamedev, html5, canvas, javascript, game-design, web-audio]
triggers:
  - html game
  - canvas game
  - browser game
  - game development html
  - shoot em up
  - brotato
  - game balance
  - game progression
  - web audio sounds
---

# HTML5 Canvas Game Development

Patterns and systems for building single-file HTML5 canvas games.
Everything runs in one .html file — no build tools, no dependencies.

---

## 1. Single-File Architecture

A complete game in one HTML file follows this skeleton:

```
<!DOCTYPE html>
<html>
<head>
  <style>
    /* Overlay screens: menu, shop, pause, death, victory */
    /* Each is position:absolute, display:none, toggled via .visible class */
  </style>
</head>
<body>
  <!-- HUD elements (health, coins, wave, weapons) -->
  <!-- Overlay divs (shop, pause, gameOver, victory, breedSelect) -->
  <!-- Canvas element -->

  <script>
    // 1. CONSTANTS & DEFINITIONS (weapons, enemies, perks, breeds)
    // 2. GAME STATE (variables, player object, arrays for entities)
    // 3. SOUND SYSTEM (Web Audio API synth)
    // 4. CORE FUNCTIONS (spawn, shoot, applyPerk, shop logic)
    // 5. UPDATE LOOP (movement, collision, projectiles, waves)
    // 6. DRAW LOOP (render everything on canvas)
    // 7. INPUT HANDLING (keys, mouse, pause)
    // 8. GAME FLOW (startWave, openShop, victory, gameOver, reset)
  </script>
</body>
</html>
```

GAME STATE MACHINE:
```
breedSelect → playing → shop → playing → ... → boss → victory
                ↓                                        ↓
             gameOver                              startNewRun
                ↓                                        ↓
           restartGame                             breedSelect
```

GAME LOOP PATTERN:
```javascript
function update(timestamp) {
    const dt = (timestamp - lastTime) / 1000; // delta time in seconds
    lastTime = timestamp;

    if (gameState === 'playing' && !paused) {
        // Update player movement
        // Update weapon firing
        // Update projectiles (movement, collision, special behaviors)
        // Update enemies (AI, attacks, mutations)
        // Update particles, fire zones, damage texts
        // Wave logic (spawning, completion check)
        // Boss logic
    }

    draw(); // Always render, even when paused
    gameLoopId = requestAnimationFrame(update);
}
```

KEY PRINCIPLE: Use dt (delta time) for ALL movement and timers.
Never assume a fixed frame rate. `entity.x += speed * dt` not `entity.x += speed`.

---

## 2. Projectile Systems

Every projectile is an object in a `projectiles` array, filtered each frame.

BASE PROJECTILE:
```javascript
{
    x, y,           // position
    vx, vy,         // velocity
    damage,          // base damage (multiplied by player.damage)
    range,           // max travel distance
    distance: 0,     // distance traveled so far
    radius,          // collision size
    pierce,          // enemies to pass through (0 = hit one and die)
    bounce,          // wall bounces remaining
    knockback,       // push force on hit
    slow,            // slow duration applied to enemy
    angle,           // current facing angle
    spin: 0,         // rotation for visual
    age: 0,          // time alive (for visual effects)
    weaponType,      // string key for rendering + special behavior
    explosive,       // splash damage amount
    expSize,         // splash radius
}
```

SPECIAL BEHAVIORS (checked in projectile update loop):

| Mechanic | How it works |
|----------|-------------|
| Pierce | `p.pierce--` on hit. Remove when `pierce < 0` |
| Bounce | On wall hit: flip vx/vy, `p.bounce--`, can add +20% damage |
| Homing | Steer toward nearest enemy: `angle += diff * turnRate` |
| Guided | Same as homing but stronger turn rate + targets boss as fallback |
| Gravity | `p.vy += gravity * dt` (cannon, coin toss arcs) |
| Acceleration | `speed = min(speed * (1 + rate * dt), maxSpeed)` |
| Damage Ramp | `p.damage += rampAmount` on each pierce (Piercer) |
| Boomerang | Decelerates, then reverses toward player. Despawns when caught |
| Ricochet | On hit, redirect velocity toward nearest OTHER enemy within range |
| Fire Zone | On impact, create persistent AoE damage area |
| Spin-up | Track `weapon.spinUp`, ramp fire rate from 30%→100% over time |

BOOMERANG PATTERN:
```javascript
if (p.weaponType === 'boomerang') {
    if (!p.returning && p.distance > p.range * 0.5) p.returning = true;
    if (p.returning) {
        // Steer back to player
        const dx = player.x - p.x, dy = player.y - p.y;
        const d = Math.sqrt(dx*dx + dy*dy);
        if (d < 25) { p.pierce = -1; } // caught — despawn
        else {
            p.vx = (dx/d) * returnSpeed;
            p.vy = (dy/d) * returnSpeed;
            p.distance -= returnSpeed * dt; // keep alive during return
        }
    } else {
        p.vx *= 0.97; p.vy *= 0.97; // decelerate outward
    }
}
```

RICOCHET PATTERN (perk-based, applies to all weapons):
```javascript
// After enemy takes damage from projectile:
if (player.ricochet && p.pierce >= 0) {
    let nearest = null, bestDist = 200;
    enemies.forEach(e => {
        if (e === hitEnemy) return; // skip the one we just hit
        const d = dist(e, p);
        if (d < bestDist) { bestDist = d; nearest = e; }
    });
    if (nearest) {
        const a = Math.atan2(nearest.y - p.y, nearest.x - p.x);
        const spd = Math.sqrt(p.vx*p.vx + p.vy*p.vy);
        p.vx = Math.cos(a) * spd;
        p.vy = Math.sin(a) * spd;
    }
}
```

---

## 3. Enemy AI Patterns

Enemies are objects in an `enemies` array. Each has a `type` that
determines behavior in the update loop.

COMMON PATTERNS:
- **Melee (paper)**: Walk toward player, stop at meleeRange, attack on timer
- **Ranged (fomo)**: Approach to preferred range, back off if too close, shoot projectiles
- **Rush (splitter/splitterling)**: Wobble movement with `sin()` perpendicular offset, speed burst when close
- **Explosive**: State machine — approach → windup (telegraph) → dash (dodgeable)
- **Tank**: Slow, high HP, high knockback resist, high damage
- **Elite**: Boosted stats of normal enemy

SPLITTER PATTERN: On death, spawn N smaller enemies:
```javascript
if (enemy.type === 'splitter' && enemy.health <= 0) {
    for (let i = 0; i < 3; i++) {
        const angle = (i / 3) * Math.PI * 2;
        pendingSpawns.push({
            x: enemy.x + Math.cos(angle) * 18,
            y: enemy.y + Math.sin(angle) * 18,
            type: 'splitterling', radius: 10, speed: 185,
            health: parent.damage * 0.6, ...
        });
    }
}
// Apply pendingSpawns AFTER the filter loop (can't modify array during filter)
```

COLLISION FIX — enemies clipping inside player:
```javascript
// Push enemies out if they overlap the player body
const minDist = player.radius + enemy.radius;
if (dist < minDist && !isRanged) {
    const pushStr = Math.min(minDist - dist, 200 * dt);
    enemy.x -= dirX * pushStr;
    enemy.y -= dirY * pushStr;
}
// Knockback on melee contact
if (dist <= meleeRange && enemy.attackTimer near 0) {
    enemy.x -= dirX * 60 * dt;
    enemy.y -= dirY * 60 * dt;
}
```

---

## 4. Mathematical Progression Engine

Five mathematical systems that create organic, non-linear game feel:

### Golden Ratio (φ = 1.618...) — Economy Scaling
Cost of repeatable purchases. Feels fair because φ appears in nature.
```javascript
const PHI = 1.618033988749895;
function goldenCost(baseCost, purchaseCount) {
    return Math.round(baseCost * Math.pow(PHI, purchaseCount));
}
// 80 → 129 → 209 → 338 → 547 → 885 ...
```

### Sigmoid Bell Curve — Spawn Timing
Few enemies → BURST → few. The derivative of sigmoid is a bell curve.
```javascript
function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function spawnBellCurve(t) { // t = 0..1 (wave progress)
    const x = (t - 0.5) * 12; // map to -6..+6
    const s = sigmoid(x);
    return s * (1 - s) * 4; // bell curve, peak ≈ 1 at t=0.5
}
// Use: spawnInterval = baseInterval * (1 / spawnBellCurve(progress))
```

### Logistic Map — Chaos at High Difficulty
`x_next = r * x * (1 - x)` — deterministic chaos from one equation.
r < 3: stable/predictable. r > 3.57: chaotic.
```javascript
function logisticMap(x, r) { return r * x * (1 - x); }
// In spawn logic:
const chaosR = Math.min(3.95, 2.5 + difficulty * 0.18);
chaosX = logisticMap(chaosX, chaosR);
const chaosFactor = 0.6 + chaosX * 0.8; // perturb spawn timing/count
```

### Prime Factorization — Mutation System
Each prime number maps to a mutation type. The prime factorization of
the difficulty determines which mutations are active and at what power.
```javascript
function primeFactorize(n) {
    const factors = {};
    for (let d = 2; d * d <= n; d++) {
        while (n % d === 0) { factors[d] = (factors[d] || 0) + 1; n /= d; }
    }
    if (n > 1) factors[n] = 1;
    return factors;
}
// Difficulty 2 (prime): Swarm mutation introduced
// Difficulty 6 (2×3): Swarm + Armored combo
// Difficulty 8 (2³): Swarm at power 3
// Every difficulty has a unique mutation fingerprint
```

Mutation examples:
- 2: Swarm (+25% spawn count per power)
- 3: Armored (enemies take less damage)
- 5: Volatile (damage zones on enemy death)
- 7: Pack Hunter (enemies get +damage near allies)
- 11: Regenerating (enemies regen HP)
- 13: Phasing (enemies teleport occasionally)

### Harmonic Series — Diminishing Returns
```javascript
function harmonicValue(baseValue, purchaseCount) {
    return baseValue / (1 + purchaseCount);
}
// 1st buy: full value. 5th: 1/5th. 20th: 1/20th.
// Always gives SOMETHING but can never break the game.
```

COMBINED: Golden ratio drives costs up, harmonic series drives
value down. Together they create a natural soft cap where buying
more is always possible but increasingly inefficient.

---

## 5. Web Audio Synth Sound Recipes

No audio files needed. All sounds created with oscillator + gain envelope.

```javascript
function playSound(type) {
    if (!audioCtx) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    const t = audioCtx.currentTime;

    switch(type) {
        case 'shoot':      // Quick blip: square 700→200 in 0.06s
        case 'explosion':  // Deep sweep: sawtooth 200→20 in 0.3s, louder
        case 'hurt':       // Short groan: sawtooth 200→100 in 0.15s
        case 'coin':       // Bright ding: sine 1200→1500 in 0.08s
        case 'kill':       // Rising chirp: sine 300→700 in 0.1s
        case 'crit':       // Ascending triplet: square 600→900→1200
        case 'waveComplete': // Victory jingle: sine 500→700→900 ascending
        case 'gameOver':   // Sad descend: sine 400→100 in 0.5s
        case 'buy':        // Cash register: sine 800→1000→1200 ascending
        case 'bossExplode': // Deep rumble: sawtooth 120→15 in 1.2s, loud
        case 'victoryFanfare': // Triumphant: sine C5→E5→G5→C6 ascending
        case 'sadDeath':   // Slow minor: triangle A4→F4→D4→fade in 1s
    }
    osc.start(t); osc.stop(t + duration);
}
```

PATTERN: osc.type sets timbre (sine=pure, square=retro, sawtooth=harsh,
triangle=soft). Frequency sweep = pitch change. Gain envelope = volume shape.

SOUND DESIGN RULES OF THUMB:
- Positive events: ASCENDING frequency (coins, kills, victory)
- Negative events: DESCENDING frequency (hurt, death, game over)
- Impact events: Start LOUD, decay fast (explosions, hits)
- Short for frequent sounds (0.03-0.1s), longer for rare (0.3-1s)
- Throttle rapid sounds (min gap per type) to avoid audio overload

---

## 6. Shop / Perk / Upgrade Systems

SHOP FLOW:
1. Wave ends (all enemies dead) → `openShop()`
2. Roll random weapon offers + perk offers (weighted by wave/luck)
3. Player buys, locks, refreshes, sells, or continues
4. `nextWave()` → increment wave → `startWave()`

PERK GATING: Higher rank perks unlock at later waves.
```javascript
if (perk.rank === 1) return currentWave >= 1;
if (perk.rank === 2) return currentWave >= 3;
// ...etc
```

DUPLICATE PROTECTION: `ownedPerkKeys` Set tracks bought perks.
Each unique perk can only be purchased once.

REPEATABLE PERKS (for endless progression):
When all unique perks are exhausted, offer stackable perks with:
- Golden ratio cost scaling (gets expensive)
- Harmonic diminishing returns (gives less each time)
- Always available — shop never goes empty

PERK CARRYOVER: On victory, carry perks + owned keys to next difficulty run.
On restart (death), wipe everything.

PITY SYSTEM: Track consecutive shops without high-rank offerings.
After N low shops, force-insert a high-rank perk.

---

## 7. Visual Juice

PARTICLES: Simple {x, y, vx, vy, life, color} objects. Spawn bursts on:
- Enemy death (8 particles, random directions)
- Boss explosions (40 particles, mixed colors)
- Projectile impacts

SCREEN SHAKE: Set `screenShake = intensity`, decay each frame:
```javascript
if (screenShake > 0.5) screenShake *= 0.9; else screenShake = 0;
// In draw: ctx.translate(random * shake, random * shake)
```

FLOATING DAMAGE NUMBERS: {x, y, amount, vy, life, color, isCrit}
Rise upward and fade. Crits are larger with different color.

CSS OVERLAY SCREENS: Use distinct visual identities:
- Death: dark red vignette, pulsing blood-red text, drip animations
- Victory: gold radial gradient, ascending glow, rise-up stagger
- Animation keyframes: fadeIn, pulse, drip, riseUp for staggered reveals

PROJECTILE RENDERING: Branch on `p.weaponType` in draw loop.
Each weapon type gets its own canvas drawing code (rockets, moons,
shurikens, coins, boomerangs, bullets, energy bolts, etc).

---

## 8. Balance Methodology

DPS CALCULATION:
```
Base DPS = damage × fireRate
Effective DPS = Base DPS + (splash × splashHitsPerShot × fireRate)
                + (fireZoneDPS × fireZoneDuration × fireRate)
```

Multiply by pierce factor for group DPS. Account for pellets.

UPGRADE SCALING: Per level, multiply damage by 1.25 and fireRate by 1.12.
Calculate DPS at max level to find outliers.

RED FLAGS:
- Any weapon > 2x DPS of same-tier weapons = needs nerf
- Any weapon with zero downsides AND top DPS = broken
- Tier 4 weapon should feel noticeably stronger than Tier 2
- Movement penalties, short range, slow fire = valid tradeoffs
- "Fun but weak" is better than "boring but strong"

DIFFICULTY SCALING: Don't just multiply one stat.
- HP scales linearly with difficulty (main scaling lever)
- Speed scales logarithmically: `speed * (1 + Math.log(diff) * 0.08)`
- Damage scales logarithmically: `damage * (1 + Math.log(diff) * 0.12)`
- Logarithmic prevents runaway scaling while still feeling harder

---

## 9. Doge Meme Integration (BrotherDoge-specific pattern)

Name every weapon/item with a doge-speak first word:
Much, Very, Wow, Such, So, Doge, Many, Amaze, Excite, HODL, Moon

HUD shows first word of each equipped weapon. 4 weapon slots = 4-word phrase.
On victory, concatenate into a meme "tweet" the player can screenshot/share.

This turns loadout selection into a creative/comedic expression system.
Players will choose weapons partly for the meme, which is good design.
