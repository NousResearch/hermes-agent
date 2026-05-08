# Hermes AI Office — DeskRPG Reference Notes

Last updated: 2026-05-09 KST
Status: research/design reference only; no dependency or code adoption approved.

## Sources inspected

- https://github.com/dandacompany/deskrpg
- https://deskrpg.com
- Public GitHub README screenshots/assets:
  - `public/readme/dantelabs-office-map.png`
  - `public/readme/home-screenshot.png`
  - `public/readme/login-home.png`

## High-level finding

DeskRPG is the closest current reference for the direction the Hermes AI Office should study next: a lightweight 2D pixel-art virtual office rather than a 3D metaverse. The product framing is “2D pixel art multiplayer virtual office game,” and the visible implementation uses a browser app with Phaser, Socket.IO, tile maps, LPC-style characters, NPCs, task loops, meeting rooms, and a browser map editor.

Hermes should not copy DeskRPG code/assets without a separate license/reuse review. The useful value is conceptual and visual: use a readable, warm, 2D office map to make AI agents feel spatially present without sacrificing dashboard clarity.

## Visual style to study

- 2D top-down / RPG Maker-like camera, not 3D.
- Tile-based office floor with warm wood flooring.
- Functional zones defined by floor materials and furniture clusters rather than heavy walls.
- Small pixel avatars with name labels.
- Desks, monitors, meeting table, lounge, utility area, whiteboard, bookshelf, water cooler, printer, plants.
- Thin top HUD with connection/player/NPC/task/meeting status.
- Small minimap for spatial orientation.
- Contextual interaction prompt near the bottom.
- Modern dark UI panels layered over a playful pixel world.

## Why this fits Hermes better than 3D

- Easier to read during actual operations.
- Lower performance and dependency burden than a 3D scene.
- Easier to keep accessible and keyboard/screen-reader fallbacks available.
- Easier to integrate with the existing read-only `/office` dashboard and Safe inspector.
- Makes agent/task state visible without introducing a distracting immersive camera.

## Hermes adaptation direction

Start with a Hermes-native 2D office projection over the existing redacted `OfficeState` DTO:

- Lobby / Entry: recent sessions and current operator context.
- Workbench: Kanban/work items as desks/cards.
- Machine room: cron/automation as machines or bots.
- Routing room: Telegram/topic/provenance as mailroom or dispatch board.
- Meeting room: future multi-agent discussion summaries, read-only first.
- Library: future NAS/Obsidian/knowledge references, read-only first.
- Unknown bucket: unresolved provenance must remain visible instead of hidden.

Possible object metaphors:

- AI agent = seated/standing avatar at a desk.
- Active task = glowing monitor or card on desk.
- Blocked task = red icon/barrier above desk.
- Cron job = machine with status light.
- Failed automation = red warning light.
- Provenance flow = floor path, arrow, conveyor, or mail route.
- Knowledge source = bookshelf/archive cabinet.
- Safe inspector = right-side serious dashboard panel, not an in-world speech bubble.

## Guardrails

- Keep the current read-only/privacy boundary.
- Pixel/2D renderer must consume only redacted `OfficeState` or a derived `PixelSceneModel`.
- Do not expose raw prompts, transcripts, task bodies, cron scripts, logs, auth, or secrets.
- Do not add mutation controls with the visual layer.
- Do not add Phaser/PixiJS or copy DeskRPG assets before dependency, security, and license review.
- Maintain a non-pixel dashboard/list fallback.
- Respect reduced-motion and small-screen layouts.

## Recommended next design step

Before adding a real Phaser/PixiJS dependency, create a dependency-free Hermes 2D style prototype inside the existing CSS/SVG map:

1. Turn the current four room nodes into a tile-like office floor sketch.
2. Add small CSS/SVG pixel-avatar placeholders for sessions/agents.
3. Use furniture-like symbols for work, automation, and routing.
4. Keep object clicks wired only to Safe inspector metadata.
5. Add fixtures for empty/error/partial/missing states.
6. Reassess whether Phaser is needed only after this prototype proves the 2D metaphor is useful.
