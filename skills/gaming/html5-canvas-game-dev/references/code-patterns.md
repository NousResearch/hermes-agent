# Code Patterns Reference

Detailed code snippets for HTML5 canvas game systems.
Load this when implementing specific mechanics.

## Projectile Specials

### Knockback Cap (prevents shotgun pellet teleport)
```javascript
enemy._frameKB = 0; // reset each frame
if (kbStrength > enemy._frameKB) {
    if (enemy._frameKB) { enemy.x -= enemy._frameKBx; enemy.y -= enemy._frameKBy; }
    enemy.x += kbX; enemy.y += kbY;
    enemy._frameKB = kbStrength;
    enemy._frameKBx = kbX; enemy._frameKBy = kbY;
}
```

### Boomerang
```javascript
if (!p.returning && p.distance > p.range * 0.5) p.returning = true;
if (p.returning) {
    const dx = player.x - p.x, dy = player.y - p.y;
    const d = Math.sqrt(dx*dx + dy*dy);
    if (d < 25) { p.pierce = -1; playSound('boomerangCatch'); }
    else {
        p.vx = (dx/d) * returnSpeed;
        p.vy = (dy/d) * returnSpeed;
        p.distance -= returnSpeed * dt;
    }
} else { p.vx *= 0.97; p.vy *= 0.97; }
```

### Ricochet
```javascript
if (player.ricochet && p.pierce >= 0) {
    let nearest = null, bestDist = 200;
    enemies.forEach(e => {
        if (e === hitEnemy) return;
        const d = dist(e, p);
        if (d < bestDist) { bestDist = d; nearest = e; }
    });
    if (nearest) {
        const a = Math.atan2(nearest.y - p.y, nearest.x - p.x);
        const spd = Math.sqrt(p.vx*p.vx + p.vy*p.vy);
        p.vx = Math.cos(a) * spd; p.vy = Math.sin(a) * spd;
    }
}
```

## Math Systems

### Golden Ratio Cost Scaling
```javascript
const PHI = 1.618033988749895;
function goldenCost(baseCost, count) {
    return Math.round(baseCost * Math.pow(PHI, count));
}
```

### Sinusoidal Wave Spawn Curve (Wave N = N surges)
```javascript
function waveSpawnCurve(t, waveNum) {
    const n = Math.max(1, waveNum);
    const raw = Math.pow(Math.abs(Math.sin(n * Math.PI * t)), 3);
    const floor = 0.05;
    return floor + raw * (1 - floor);
}
// Use time-based progress: waveTimer / waveDuration
```

### Logistic Map (chaos at high difficulty)
```javascript
function logisticMap(x, r) { return r * x * (1 - x); }
const chaosR = Math.min(3.95, 2.5 + difficulty * 0.18);
chaosX = logisticMap(chaosX, chaosR);
```

### Prime Factorization (mutation fingerprints)
```javascript
function primeFactorize(n) {
    const factors = {};
    for (let d = 2; d * d <= n; d++) {
        while (n % d === 0) { factors[d] = (factors[d] || 0) + 1; n /= d; }
    }
    if (n > 1) factors[n] = 1;
    return factors;
}
// 2=Swarm, 3=Armored, 5=Volatile, 7=PackHunter, 11=Regen, 13=Phasing
```

### Harmonic Diminishing Returns
```javascript
function harmonicValue(baseValue, count) {
    return baseValue / (1 + count);
}
```

## Perk Implementation Patterns

### Fire Rate Bonus
```javascript
if (player.fireRateBonus > 0) fireRate *= (1 + player.fireRateBonus);
```

### Lifesteal
```javascript
if (player.lifesteal > 0) {
    player.health = Math.min(player.maxHealth, player.health + damage * player.lifesteal);
}
```

### Armor + Thorns (in damage-receiving code)
```javascript
if (player.armor > 0) damage = Math.max(1, damage - player.armor);
player.health -= damage;
if (player.thorns > 0) { enemy.health -= player.thorns; }
```

### Leverage (conditional damage at projectile hit)
```javascript
if (player.leverage) {
    const ratio = player.health / player.maxHealth;
    if (ratio > 0.8) damage *= 1.6;
    else if (ratio < 0.4) damage *= 0.8;
}
```

### Triggered Shield (FOMO Shield)
```javascript
// In update loop:
if (player.fomoShield && player.fomoShieldHP <= 0 && player.fomoShieldCooldown <= 0) {
    if (player.health < player.maxHealth * 0.25) {
        player.fomoShieldHP = 50; player.fomoShieldCooldown = 60;
    }
}
// In damage code: absorb from shield HP before player HP
```

### Immediate Trade Perk (HP for coins)
```javascript
const hpLoss = Math.floor(player.maxHealth * 0.1);
player.maxHealth -= hpLoss;
if (player.health > player.maxHealth) player.health = player.maxHealth;
coins += 50;
spawnDamageText(player.x, player.y - 30, '+50🪙', '#ffd700', false);
spawnDamageText(player.x, player.y - 10, `-${hpLoss}❤️`, '#ff4444', false);
```

### Shop Discount
```javascript
const discount = 1 - (player.shopDiscount || 0);
return Math.round(basePrice * discount);
```

### Passive Coin Generation
```javascript
if (player.coinGen > 0) {
    player.coinGenTimer += dt;
    if (player.coinGenTimer >= 3) { player.coinGenTimer -= 3; coins += player.coinGen; }
}
```

## Sticker/Badge Perk Display (collage mode at 8+ perks)
```javascript
const isCollage = player.perks.length >= 8;
hud.className = isCollage ? 'collage' : 'clean';
entries.forEach(([label, data], i) => {
    let style = '';
    if (isCollage) {
        const seed = (i * 7 + data.rank * 13) % 30;
        const rotation = (seed - 15) * 1.2;
        const scale = 0.9 + (data.rank / 5) * 0.2;
        const zIndex = data.rank * 10 + (entries.length - i);
        style = `transform:rotate(${rotation}deg) scale(${scale}); z-index:${zIndex};`;
    }
    html += `<div class="perk-sticker rank-${data.rank}" style="${style}">${label}</div>`;
});
```

## Roguelite Carryover (victory only, death = lose all)
```javascript
// In victory():
pendingCarryCoins = Math.ceil(coins * 0.1);
pendingCarryWeapon = findWeakestWeapon(); // lowest tier, then lowest level, reset to lvl 1
// In selectBreed():
if (pendingCarryWeapon) player.weapons = [pendingCarryWeapon]; else defaultWeapon;
if (pendingCarryCoins > 0) coins = pendingCarryCoins;
// In restartGame() (death): clear both + reset difficulty
```

## Worms Artillery Patterns

### Trajectory with Wind
```javascript
p.vx += wind * windStrength * dt;
p.vy += gravity * dt;
p.x += p.vx * dt;
p.y += p.vy * dt;
```

### Charge-to-Fire
```javascript
// Space down: startCharge = Date.now()
// Space up:
const power = Math.min(1, (Date.now() - startCharge) / 3000);
const speed = 200 + power * 800;
// Launch at aimAngle with speed
```

### Destructible Terrain (heightmap)
```javascript
function destroyTerrain(cx, cy, radius) {
    for (let x = cx - radius; x <= cx + radius; x++) {
        const dist = Math.abs(x - cx);
        const depth = Math.sqrt(radius * radius - dist * dist);
        terrain[x] = Math.max(terrain[x], cy + depth);
    }
}
```

## Sound Design

### Pitch-Scaling Coin Pickups
```javascript
let coinStreakCount = 0, coinStreakTimer = 0;
// On coin pickup:
coinStreakCount++;
coinStreakTimer = 0.4;
const pitchBoost = Math.min(400, coinStreakCount * 30);
osc.frequency.setValueAtTime(1200 + pitchBoost, t);
// In update: if (coinStreakTimer <= 0) coinStreakCount = 0;
```

### Easter Egg Key Buffer
```javascript
let secretBuffer = '';
// In keydown during specific gameState:
secretBuffer += e.key.toLowerCase();
if (secretBuffer.length > 10) secretBuffer = secretBuffer.slice(-10);
if (secretBuffer.endsWith('secretword')) { /* activate */ }
```
