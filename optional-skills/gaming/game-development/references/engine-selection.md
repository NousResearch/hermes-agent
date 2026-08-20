# Engine Selection

Choosing the engine is the highest-leverage early decision. The wrong choice
costs months. Match the engine to **the game**, **the team size**, and **the
developer's experience** — not to hype.

---

## The Decision Matrix

| Factor | Godot | Unity | Unreal | Web (JS/TS) |
|---|---|---|---|---|
| **Best for** | 2D, small 3D, indie | 3D + 2D, mobile, indie→mid | High-end 3D, AAA | Tiny/web games, jams |
| **Language** | GDScript, C#, C++ | C# | C++, Blueprints | JS/TS |
| **Learning curve** | Gentle | Moderate | Steep | Gentle (if you know web) |
| **Cost** | Free, MIT, no royalties | Free tier; fees at revenue | 5% royalty over $1M | Free |
| **2D support** | Excellent (first-class) | Good | Workable, not its focus | Good (canvas/WebGL) |
| **3D support** | Good, improving fast | Very good | Best-in-class | Limited (three.js) |
| **Mobile export** | Good | Excellent | Heavy builds | PWA only |
| **Asset ecosystem** | Growing | Massive (Asset Store) | Large (Marketplace) | npm libs |
| **Editor weight** | ~100 MB, instant | ~GBs, heavier | ~100+ GB, heavy | None (any editor) |
| **Job market** | Smaller | Large | Large (AAA) | Web-general |

---

## Recommendations by Scenario

**First game ever, solo, want to finish something:**
→ **Godot** (2D) or **Unity** (3D). Godot's lightweight editor and gentle
curve make finishing more likely. Avoid Unreal as a first engine — its power
comes with complexity that buries beginners.

**2D game (platformer, puzzle, roguelike, RPG):**
→ **Godot.** First-class 2D, no royalties, fast iteration. Unity is fine too
but heavier than you need for pure 2D.

**3D indie game, want max tutorials + asset store:**
→ **Unity.** The ecosystem is unmatched for solo/small 3D — you can buy or find
almost any system, and tutorials are everywhere.

**Photorealistic / AAA-fidelity 3D, larger team:**
→ **Unreal.** Nanite, Lumen, and the rendering pipeline are the best available.
Worth the steep curve only when visuals are the point and you have the team.

**Game jam / tiny web game / portfolio piece:**
→ **Web (HTML5 canvas + JS/TS)**, or a light framework (Phaser for 2D,
three.js for 3D). Instant load, shareable by URL, full control, great for
learning fundamentals.

**Pure mechanics prototype, no art yet:**
→ **Python + Pygame.** Fastest path to testing whether a mechanic is *fun*
before committing to an engine. Throw it away after.

---

## Common Mistakes

1. **Picking Unreal because it looks the most powerful.** Power you can't wield
   is a liability. Most indie games never need Nanite/Lumen.
2. **Picking by language familiarity alone.** Knowing C# isn't a reason to use
   Unity for a 2D pixel platformer where Godot would ship faster.
3. **Engine-hopping.** Switching engines mid-project resets you to zero. Choose
   deliberately, then commit. The engine is rarely why a game fails — scope is.
4. **Building a custom engine for a first game.** Unless engine-building *is*
   the goal, use an existing one. You'll spend years on tooling instead of the
   game.

---

## After Choosing

- Install via the official path (Godot: direct download, no account; Unity:
  Unity Hub; Unreal: Epic Games Launcher).
- Grab the canonical `.gitignore` for that engine from
  `github.com/github/gitignore` (`Unity.gitignore`, `UnrealEngine.gitignore`,
  `Godot.gitignore`).
- Set up Git LFS for binary assets before the first commit.
- Build the engine's "hello world" (move a sprite/cube) to confirm the
  toolchain works end-to-end before real work starts.
