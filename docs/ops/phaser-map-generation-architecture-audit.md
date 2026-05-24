# Phaser Map Generation Architecture Audit

This audit answers whether Swagbound interiors should be generated as concept
art, Phaser tilemaps, Tiled JSON, Phaser Editor scenes, or a hybrid pipeline.
It is intentionally strict: generated art can accelerate review, but runtime
map data must be deterministic, inspectable, validated, and explicitly
promoted.

## Verdict

Swagbound interiors should use a hybrid pipeline:

1. Generate review-only concepts, prompts, and layout capsules.
2. Promote only approved designs into Phaser-native map data.
3. Store runtime interiors as Tiled JSON or Phaser Editor tilemaps that compile
   through Phaser's Tilemap APIs.
4. Represent gameplay semantics through typed object layers and explicit
   metadata, not by interpreting generated pixels.

Do not ship 70 interiors as runtime raster backgrounds. A raster can be a
visual reference, proof image, or a baked decorative layer after approval, but
collision, exits, NPCs, service counters, and story triggers need structured
map data.

## Phaser Docs Findings

Official Phaser and Tiled docs support the hybrid route:

- Phaser `Tilemap` is a data container, not a display object. It can parse
  Tiled JSON, CSV, or 2D arrays, while tilemap layers are the rendered display
  objects.
- Phaser loads Tiled JSON maps through `this.load.tilemapTiledJSON(...)` in
  preload and uses the loaded map in create.
- `createLayer(...)` addresses layers by the Tiled layer name, which means layer
  naming is part of the runtime contract.
- Phaser Tilemap supports Tiled object layers through object-layer APIs such as
  object filtering and lookup, so doors, NPC spawns, counters, and interactables
  belong in object layers.
- Phaser parses Tiled custom tile properties into runtime tile data. Collision
  can therefore use tile properties such as `collides: true`, plus explicit
  shape objects where grid collision is too coarse.
- Phaser Editor v5 supports Tiled JSON maps and editable tilemaps. Its editable
  tilemap compiler stores map data in Tiled format and then creates the map via
  the normal Phaser Tilemap API.
- Tiled JSON has native `tilelayer`, `objectgroup`, `imagelayer`, and `group`
  layer types, object geometry, and custom properties. That is the right place
  for authorable runtime metadata.
- Phaser camera bounds are explicit rectangles via `camera.setBounds(x, y,
  width, height)`. Camera limits should come from the approved map dimensions
  or an explicit override in the interior contract.

References:

- Phaser Tilemap API: https://docs.phaser.io/api-documentation/class/tilemaps-tilemap
- Phaser Loader `tilemapTiledJSON`: https://docs.phaser.io/api-documentation/class/loader-loaderplugin
- Phaser cameras: https://docs.phaser.io/phaser/concepts/cameras
- Phaser Editor tilemaps: https://docs.phaser.io/phaser-editor/scene-editor/game-objects/tilemap-object
- Phaser Editor editable tilemaps: https://docs.phaser.io/phaser-editor/scene-editor/game-objects/editable-tilemap-object
- Tiled JSON format: https://doc.mapeditor.org/en/stable/reference/json-map-format/
- Tiled custom properties: https://doc.mapeditor.org/en/stable/manual/custom-properties/

## Current Swagbound Pipeline

The current local asset-lab shape is useful as a review pipeline:

- Raw generated sheets are processed into transparent PNG candidates.
- Review pages and contact sheets are produced for human inspection.
- Manifests record candidate IDs, intended scenes, asset roles, dimensions,
  suggested collision footprints, rough walkable zones, and collision notes.
- Building batches can group exterior art, interior core art, props, and NPC
  sheets.

That is not yet a Phaser runtime pipeline:

- Candidate PNGs are not tilemaps.
- Suggested collision footprints are not validated collision.
- Walkable zones are not proven pathing data.
- Intended scenes are not door-target contracts.
- Contact sheets are review evidence, not runtime scenes.
- There is no canonical interior schema tying building ID, door ID, map
  dimensions, object layers, collisions, story flags, and proof screenshots
  together.

## Answers

1. Swagbound interiors should be a hybrid: review-only concepts and capsules
   first, then Tiled JSON or Phaser Editor tilemaps for runtime.
2. The 70 interiors can safely generate concept art, prompt capsules, and rough
   layout capsules now. Runtime Tiled JSON, collision specs, object-layer specs,
   door maps, and Phaser Editor scenes are blocked until the contract and
   promotion gates exist.
3. The Phaser-native contract is a versioned interior manifest plus a Tiled
   JSON or Phaser Editor tilemap representation using fixed layer and object
   naming.
4. Doors, exits, NPCs, counters, shop objects, and interactables should be Tiled
   object-layer entries with typed classes and custom properties.
5. Collision should be represented with a collision tile layer using tile
   properties, plus object-layer rectangles or polygons for non-grid blockers.
   Generated collision notes remain review-only until validated.
6. Camera bounds should be explicit contract fields, defaulting to the map
   pixel dimensions. Small rooms may use a lock or center policy, but that
   policy must be declarative.
7. Interiors link to overworld buildings through stable `buildingId`,
   `doorId`, `interiorId`, and reciprocal exit object metadata.
8. Generated candidate art becomes runtime-safe only after review, manual or
   assisted translation into structured map data, schema validation, gameplay
   validation, proof screenshots, and explicit promotion.
9. Raw rasters, contact sheets, prompts, AI-suggested collision, AI-suggested
   door mapping, and unvalidated object placement must stay review-only.
10. Promotion needs explicit approval for the interior contract, map/collision
    format, each promoted candidate batch, and any write into Phaser runtime
    paths.

## Proposed Interior Contract

Use a versioned JSON contract adjacent to the map pipeline. The runtime loader
can consume a normalized form, while Tiled or Phaser Editor remains the authoring
surface.

```ts
type PromotionStatus =
  | "review_only"
  | "contract_draft"
  | "template_ready"
  | "validated_candidate"
  | "approved_for_runtime"
  | "promoted_runtime"
  | "rejected";

interface SwagboundInteriorContract {
  schemaVersion: "swagbound.interior.v1";
  interiorId: string;
  buildingId: string;
  doorId: string;
  roomType:
    | "home"
    | "bedroom"
    | "shop"
    | "clinic"
    | "diner"
    | "library"
    | "school"
    | "arcade"
    | "motel"
    | "civic"
    | "service"
    | "story"
    | "other";

  visualSourceCandidate: {
    kitId?: string;
    candidateId?: string;
    rawPath?: string;
    processedPath?: string;
    contactSheetPath?: string;
    promptPath?: string;
    reviewNotes?: string[];
    runtimePromotionAllowed: boolean;
  };

  representation: {
    kind: "tiled_json" | "phaser_editor_scene" | "hybrid_tilemap";
    mapKey: string;
    mapPath?: string;
    scenePath?: string;
    tilesets: Array<{ key: string; imagePath: string; tileWidth: number; tileHeight: number }>;
    requiredLayers: {
      floor: string;
      walls: string;
      decorBelowActors?: string;
      decorAboveActors?: string;
      collision: string;
      objects: string;
    };
  };

  dimensions: {
    tileWidth: number;
    tileHeight: number;
    columns: number;
    rows: number;
    widthPx: number;
    heightPx: number;
  };

  cameraBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
    policy: "map_bounds" | "center_if_smaller_than_viewport" | "explicit";
  };

  spawnPoint: {
    objectId: string;
    x: number;
    y: number;
    facing: "up" | "down" | "left" | "right";
  };

  exitPoint: {
    objectId: string;
    x: number;
    y: number;
    targetMapId: string;
    targetDoorId: string;
    facing: "up" | "down" | "left" | "right";
  };

  collisionLayer: {
    layerName: string;
    tileProperty: "collides";
    objectShapeLayer?: string;
    debugRequired: boolean;
  };

  objectLayer: {
    layerName: string;
    requiredClasses: Array<
      | "spawn"
      | "exit"
      | "npc_spawn"
      | "interactable"
      | "service_counter"
      | "shop_stock"
      | "story_trigger"
    >;
  };

  interactables: Array<{
    id: string;
    objectId: string;
    kind: "inspect" | "pickup" | "phone" | "sign" | "terminal" | "container" | "trigger";
    dialogueId?: string;
    itemId?: string;
    scriptId?: string;
    requiredFlags?: string[];
    setsFlags?: string[];
  }>;

  npcPlacements: Array<{
    id: string;
    objectId: string;
    npcId: string;
    spriteKey: string;
    dialogueId: string;
    facing: "up" | "down" | "left" | "right";
    collisionRadiusPx: number;
    spawnFlags?: string[];
  }>;

  shopServiceCounters: Array<{
    id: string;
    objectId: string;
    serviceType: "shop" | "heal" | "save" | "inn" | "info" | "quest";
    inventoryId?: string;
    clerkNpcId?: string;
    interactionSide: "front" | "back" | "left" | "right";
    requiredFlags?: string[];
  }>;

  storyFlags: {
    required?: string[];
    setOnEnter?: string[];
    setOnInteract?: Record<string, string[]>;
  };

  proofScreenshots: Array<{
    label: string;
    path: string;
    viewport: { width: number; height: number };
    showsCollisionDebug?: boolean;
    showsDoorRoundTrip?: boolean;
  }>;

  validationRequirements: {
    schemaValid: boolean;
    tiledJsonParses: boolean;
    requiredLayersPresent: boolean;
    requiredObjectsPresent: boolean;
    doorTargetsResolve: boolean;
    reciprocalExitResolves: boolean;
    spawnWalkable: boolean;
    exitWalkable: boolean;
    npcPositionsWalkable: boolean;
    collisionDebugScreenshot: boolean;
    cameraBoundsVerified: boolean;
    noForbiddenPathWrites: boolean;
  };

  promotionStatus: PromotionStatus;
  approvals: Array<{
    gate: "contract" | "template" | "art" | "map" | "collision" | "door_mapping" | "runtime_promotion";
    approvedBy: string;
    approvedAt: string;
    notes?: string;
  }>;
}
```

## Runtime Layer Contract

Recommended layer names:

| Layer | Type | Runtime use |
| --- | --- | --- |
| `floor` | tilelayer | Walkable floor and base visual tiles |
| `walls` | tilelayer | Wall visual tiles |
| `decor_below_actors` | tilelayer or objectgroup | Rugs, low props, floor clutter |
| `collision` | tilelayer | Blocking tile data, usually hidden in production |
| `objects` | objectgroup | Spawns, exits, NPCs, counters, story triggers |
| `decor_above_actors` | tilelayer or objectgroup | Overhead pieces rendered above actors |

Recommended object classes and properties:

| Class | Required properties | Notes |
| --- | --- | --- |
| `spawn` | `spawnId`, `facing` | Interior entry spawn from overworld door |
| `exit` | `doorId`, `targetMapId`, `targetDoorId`, `facing` | Reciprocal link back to overworld or another room |
| `npc_spawn` | `npcId`, `spriteKey`, `dialogueId`, `facing` | Can include `spawnFlags` and schedule metadata |
| `interactable` | `interactableId`, `kind` | Inspect, pickup, sign, phone, terminal, container |
| `service_counter` | `serviceType`, `interactionSide` | Shop, heal, save, inn, info, quest |
| `shop_stock` | `inventoryId` | Optional separate stock trigger |
| `story_trigger` | `triggerId`, `requiredFlags`, `setsFlags` | Must be testable |

## Collision Contract

Collision should not be inferred from the final rendered art. Use:

- `collision` tile layer with tiles whose tileset metadata includes
  `collides: true`.
- Optional `collision_shapes` object layer for rectangles or polygons when a
  prop, counter, wall curve, or display case does not align to the tile grid.
- Runtime validation that spawn, exit, NPC, and interaction approach points are
  walkable.
- A proof screenshot with collision debug enabled before promotion.

Generated `suggested_collision_footprint` values are useful review hints. They
are not runtime collision until converted into map data and validated.

## Door And Overworld Linking

Each overworld building door needs a stable `doorId`. The interior contract must
reference the same `buildingId` and `doorId`, and the interior `exit` object must
round-trip to the overworld target.

Minimum promotion checks:

- Every overworld `doorId` resolves to exactly one `interiorId`.
- Every interior `exit` resolves to a known target map and target door.
- The player spawn after entering and after exiting is walkable.
- Facing direction is explicit on both sides.
- Story-gated doors include required flags and locked feedback.

## Camera Bounds

Default camera bounds:

```ts
this.cameras.main.setBounds(0, 0, map.widthInPixels, map.heightInPixels);
```

The contract should still store the bounds explicitly because small interiors
may need a center-lock policy and unusual rooms may need offsets. Camera bounds
must be checked in proof screenshots so the viewport does not reveal blank map
space or crop required interaction objects.

## Candidate Art To Runtime Data

Generated candidate art becomes runtime-safe only through this promotion path:

1. Generate review-only concept or layout capsule.
2. Human review selects a candidate.
3. Translate the design into a Tiled JSON map or Phaser Editor tilemap using
   approved tilesets and layer names.
4. Add typed object-layer entries for spawns, exits, NPCs, counters, and
   interactables.
5. Add collision tile data and optional collision shapes.
6. Run schema, parse, pathing, door resolution, and camera validation.
7. Capture proof screenshots, including collision debug and door round-trip.
8. Record approvals and only then promote runtime data.

## Review-Only Assets

These must stay review-only:

- Raw generated raster sheets.
- Processed PNG candidates and contact sheets.
- Prompt text and capsule drafts.
- AI-proposed door mappings.
- AI-proposed collision footprints and walkable zones.
- AI-proposed NPC, counter, and interactable placements.
- Any generated Tiled JSON or Phaser Editor scene before schema, gameplay, and
  proof validation pass.

## Approval Gates

Promotion needs explicit approval for:

1. The canonical interior contract.
2. The Tiled or Phaser Editor authoring template.
3. The approved candidate visual direction.
4. The tilemap and object-layer data.
5. The collision layer and collision shapes.
6. The overworld door mapping.
7. The proof screenshots.
8. The final write to Phaser runtime paths.

No generated batch should write into runtime map, collision, door-target, source,
package, status, Godot, proof-map, or pro-review-packet paths without that final
promotion approval.

## 70-Interior Batch Readiness

| Output | Classification | Reason |
| --- | --- | --- |
| Concept art only | A. Safe now as review-only concept generation | It does not change runtime behavior if stored only in review areas. |
| Contact sheets and review pages | A. Safe now as review-only concept generation | Useful for human selection, not runtime data. |
| Prompt capsules | B. Safe now as layout/capsule/prompt generation | They help scope rooms without claiming runtime validity. |
| Narrative room capsules | B. Safe now as layout/capsule/prompt generation | Safe if they stay descriptive and review-only. |
| Rough dimensions and room-type inventory | B. Safe now as layout/capsule/prompt generation | Safe as planning data, blocked from runtime until contract approval. |
| Door mapping proposal | B now, D for promotion | Proposal is safe; writing door targets is blocked until approval. |
| Tilemap-ready layout spec | C. Blocked until Phaser interior contract exists | "Tilemap-ready" needs canonical layers, IDs, and validation rules first. |
| Tiled JSON map | C and D | Requires the contract and map/collision/promotion approval. |
| Phaser Editor scene | C and D | Requires the contract, editor template, and promotion approval. |
| Collision spec | D. Blocked until map/collision/promotion approval | Runtime collision changes gameplay and must be validated. |
| Object-layer spec | C. Blocked until Phaser interior contract exists | Needs stable object classes and required properties first. |
| NPC placements | C for schema, D for runtime | Placement affects traversal, story, and collision. |
| Shop/service counters | C for schema, D for runtime | Service behavior requires typed metadata and tests. |
| Runtime raster backgrounds for all 70 | E. Should not be generated | Raster-only interiors do not encode doors, collision, or interactions safely. |
| Direct writes to Phaser runtime paths | E until approved | Promotion gate is missing. |
| Godot interior artifacts | E. Should not be generated for Phaser promotion | They do not establish a Phaser-native runtime contract. |

## Recommendation

Do not generate 70 runtime interiors now.

Proceed with 70 review-only prompt capsules and, if visual review bandwidth is
available, 70 review-only concept interiors. Block all promotion. Before
generating Tiled JSON or Phaser Editor scenes, create one canonical template and
one pilot interior that proves:

- The contract is complete.
- Tiled or Phaser Editor output can be loaded by Phaser.
- Collision and object layers validate.
- Door round-trip works.
- Camera bounds render correctly.
- Proof screenshots are captured.
- Forbidden runtime paths are not touched without approval.

After the pilot passes, batch-generate structured candidates in smaller groups,
with promotion handled per interior rather than as a 70-map bulk import.

## Validation Requirements

Minimum validation before runtime promotion:

- JSON schema validates the interior contract.
- Tiled JSON parses or Phaser Editor scene compiles.
- Required layers exist with exact names.
- Required object classes and properties exist.
- Tile dimensions and pixel dimensions match the contract.
- Camera bounds are explicit and screenshot-verified.
- Spawn and exit points are walkable.
- Door targets resolve both directions.
- NPC and counter approach points are walkable.
- Collision debug proof screenshot exists.
- Story flags referenced by objects are known.
- Runtime data writes are limited to approved paths.
- Forbidden-path grep is clean until the promotion gate is approved.

## Promotion Status Flow

Recommended statuses:

1. `review_only`
2. `contract_draft`
3. `template_ready`
4. `validated_candidate`
5. `approved_for_runtime`
6. `promoted_runtime`
7. `rejected`

The default for all generated 70-interior outputs should be `review_only`.
