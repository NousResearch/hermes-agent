# Specialist worker fitting and animation checklist — 2026-09-07

This is a read-only audit of preserved cards, saved generated-mesh renders, GLB structure and
current rigging code. It changes no model, Blender scene, rigging script or runtime selection.
No generic specialist rig is accepted by this document. Baseline retains its existing review
status; its successful runtime checks do not approve the specialist family.

Audit snapshot: **2026-09-08 03:01 UTC** (2026-09-07 evening local time). All 20 cards were
visually inspected. Nineteen generated material meshes were initially available; upstream
maintenance was added and inspected after its completion at about 20:05 local time. All 20
now have card and generated-mesh inspection in this audit. Source and GLB hashes/availability are recorded in
[evidence/worker-rig-intake-audit.json](evidence/worker-rig-intake-audit.json). Queue output may
advance after this snapshot.

Every available material GLB contains one mesh node, not separate semantic mechanical parts.
The preserved high-resolution topology audits place 96.4–99.8% of faces in their largest
connected component. “Separate by loose parts” therefore cannot reliably recover a wheel,
palm, tool, fender or hose. Split/fix the semantic surfaces at real joint seams before fitting;
a valid zero-unweighted report alone does not prove the mechanics are correct.

## Fix the shared failure mechanism first

The current [worker fitter](../../../apps/desktop/scripts/lunar_worker_rig.py) gives every wheeled
variant four normalized wheel centres (`x=±0.18H`, depth `±0.12H`, height `0.09H`, lines 58–61).
It then assigns a broad low/side vertex region to those wheels (lines 109–111). Core-runtime
has stabilizer feet and braces in exactly that region; editorial has six wheels. Both violate
those assumptions. Its hand mask can also capture low props/supports (lines 105–108).

The same fitter blends weights through general mesh adjacency (lines 124–136) and parents
all extra objects to `head` (lines 140–141). Both require semantic boundaries: a welded tire/fender
or hand/tool contact must not diffuse weights, and a newly separated backpack, tray or wheel
must not follow the head. The current animation exporter gives non-biped/non-wheeled travel a
chest wobble, so track and tripod motion also remains unfinished even when `walk` exists.

- [ ] For each worker, preserve the source and create a named part inventory before modifying its rig.
- [ ] Record each wheel's centre, axle unit vector, radius, associated faces and stationary housing;
      measure in the finished mesh, not a common fraction of total height. Use front/side/rear to
      distinguish a caster, stabilizer foot, sprocket and wheel. Do not invent centres for merged hubs.
- [ ] Assign hard shells, tires, tools, panels and props to one controlling bone/rigid node per part.
      Restrict multi-bone blending to real flexible joints, hoses and cables. Mask smoothing at
      every mechanical part boundary. Recheck boundaries after welding or remeshing.
- [ ] Fit shoulder/elbow/wrist and support hinges in 3D, including depth. Pose differences between
      the card's elevations must be reconciled; nearest-distance-to-a-flat-chain is insufficient.
- [ ] Keep +Y-up, +Z-front glTF and 1 unit = 1 metre. Heights below are chosen design targets,
      not measured architecture. Maintain grounded pivots; define the hover gap separately.
- [ ] Record real auxiliary chains and props. Bone names are internal: the game clones all bones
      and transform targets and does not require every worker to have the same bone count.

## Per-worker checklist

In the table, “fit” is work still required. Named items are proposed semantic controls, not a
claim that corresponding independent parts already exist. Left/right must be defined from
the character's perspective after +Z-front alignment, not from the screenshot side.

| Worker / evidence | Locomotion | Manipulators and rigid attachments | Fitting checklist | Required animation/geometry checks |
|---|---|---|---|---|
| [`baseline`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/baseline/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/baseline/finish-front.png) | Biped; 1.20 m | Two hands; one combined face/chassis shell. | Fit hip/knee/ankle centres and rigid soles independently. Keep visor, handle and antenna on chassis; do not bend a fictional neck through its face. Separate finger shells only where grasping requires them. | Idle/walk/work/tool-use already have runtime review evidence. Preserve sole contact, grip shape and the held wait state; no extrapolation of this result to specialist rigs. |
| [`acceptance-release`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/acceptance-release/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/acceptance-release/finish-front.png) | Biped; 1.20 m | Two arms; inspection camera/probe on one hand, bulky approval panel on the other forearm. | Fit asymmetric arms in 3D. Assign hip covers to pelvis/thigh assemblies, never nearby hands. Camera, panel and handle remain rigid; wrist and gripper pivots must not include the hip skirts. | Walk with both feet planted during stance; raise inspection camera during review/triage while opposite panel and hip shells retain their shape. Existing hip-weight failure remains a rejection until rechecked. |
| [`archive-acquisition`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/archive-acquisition/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/archive-acquisition/finish-front.png) | Four-wheel chassis; 0.95 m | Two arms; handheld scanner, wrist device and side storage pouches. | Measure all four wheel hubs/radii from side views, isolate tires from fenders and pouches, and weight each wheel to its own axle. Pouches belong to torso; scanner belongs to its hand. | Walk rolls all wheels with correct sign; scanner aim changes during review without pulling side bags. Wheel/fender clearances and compact height need independent QA. |
| [`arts-studio`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/arts-studio/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/arts-studio/finish-front.png) | Three radial support/roller feet; 1.20 m | Two primary body arms plus higher utility branches around the head/shoulders; brush, small gripper and palette support. | Inventory every branch against plan/side before rigging. Add distinct auxiliary upper/forearm/terminal chains for the upper branches, and a rigid palette/support chain. The generated upper branch on one side terminates abruptly; the other approaches/fuses with a primary arm. Repair continuity first. Fit three support roots and any actual terminal rollers individually. | Work moves brush independently of palette and gripping hand; head turns cannot shear the utility branches. Tripod supports stay planted or roll with their measured axes. The two-arm generic rig cannot validate this mechanism. |
| [`ci-repair-triage`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/ci-repair-triage/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/ci-repair-triage/finish-front.png) | Two continuous tracks; 1.20 m | Two asymmetric arms: multi-probe tool and heavy pincer with hose. | Isolate track belts, housings and sprockets. Fit both arm depths and exact rigid tool shells; keep hose flexible with its own short chain. Front generated render has poorly resolved probe tips and a detached fragment: repair tool silhouette, not merely its weights. | Walk must represent tracked travel, not rotate entire track pods. Work/triage keep probes rigid and pincer/hose clear of chassis. Re-run the rejected equipment deformation check after fitting. |
| [`content-studio`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/content-studio/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/content-studio/finish-front.png) | Four-wheel chassis; 1.20 m | Two arms; camera/light bracket and pen-like tool; back storage panel. | Four independent hub centres; mask fenders out. Parent camera and light frame to one rigid wrist assembly and pen to the other; panel to torso. Resolve fused finger/tool contacts before allowing wrist twist. | Camera framing in review and pen strokes in work must be independent. Walk stops wheel rotation when stationary; camera frame cannot flex with hand smoothing. |
| [`community-intake`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/community-intake/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/community-intake/finish-front.png) | Four-wheel chassis; 1.20 m | Two arms; open greeting hand and held tablet; integrated egg-shaped upper shell. | Treat upper shell as a rigid assembly rotating at waist; avoid a neck cut through the two front displays. Four wheel centres; tablet and bezel rigid to hand. Reconstruct individual greeting fingers only as needed. | Talk/listen show an open palm without tablet warping; work brings tablet into view. Lower tablet fingers overlap wheel-height regions, so test all wheels and the hand separately. |
| [`control-plane-incidents`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/control-plane-incidents/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/control-plane-incidents/finish-front.png) | Two continuous tracks; 1.20 m | Two arms; held control tablet; rotating dome, tall rear mast, beacon and tanks. | Track pods stay rigid to chassis; sprockets get individual axes. Dome rotates at its bearing. Rear mast, tanks and piping belong to torso, not head or hands. Separate tablet from palm deformation. | Triage points/reads tablet while mast and track housings stay still. Do not let head scanning carry rear hardware. Generic chest wobble alone is not completed track animation. |
| [`core-runtime-ux-repairs`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/core-runtime-ux-repairs/finish-front.png) | Wheels plus deployable stabilizers; 1.20 m | Two tool arms with braced/parallel links, tablet and fine probe; front/rear outrigger feet and rear service pack. | Priority repair: segment four tire/hub assemblies from fenders, low braces and stabilizer pads before weights. Fit actual front/rear hub centres and each stabilizer hinge/foot separately; verify any central caster/underbody projection rather than assuming a fifth wheel. Match both links of braced forearms to their true pivots. | Walk retracts/clears stabilizers and rotates only tires/hubs. Work deploys supports, then moves probe/tablet. Side mesh visibly retains four wheels and low pads, so a global low-Z wheel mask is demonstrably wrong. Current export failure remains held. |
| [`data-performance-repairs`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/data-performance-repairs/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/data-performance-repairs/material-review.png) | Hover/base skirt; 1.20 m | Two arms with probe and pincer/hose; spherical upper shell with cooling vents. | No exposed wheels in card; generic classification is hover. Keep skirt/vent shell rigid and separate from arms, pin cooling hardware to shell. Probe remains straight; hose receives an explicit flexible chain. | Walk uses controlled hover travel/lean with documented ground clearance; never add a biped gait. Work rotates tool at wrist without bending probe or narrowing sphere. Cooling arrows are card annotations, not geometry. |
| [`engineering-guild`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/engineering-guild/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/engineering-guild/material-review.png) | Biped; 1.20 m | Two arms; large hammer, long pincer and integrated spherical upper shell. | Fit hips, knees and ankles to actual joints, with rigid armour on each segment. Hammer/head/shaft must be a rigid hand-held prop; restore a credible grip if fused with hand. Keep spherical body and antenna rigid above pelvis. | Work performs a restrained hammer arc with wrist/shaft fixed together; carry and walk must clear hammer from knees. Generated fingers/forearm edges are rough and need geometry cleanup before severe flexion tests. |
| [`editorial-desk`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/editorial-desk/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/editorial-desk/material-review.png) | Six-wheel chassis; 1.20 m | Two arms; pen-like tool and box camera; rear module. | Source side shows three wheels per side: use front/middle/rear axle bones on each side, not four. Generated side has merged/lumpy hubs and fenders; restore six distinguishable rolling parts, then fit each centre and radius. Keep camera/pen rigid to hands. | Walk checks all six hubs simultaneously and verifies fenders remain stationary. Work uses pen while camera remains stable. Middle wheels must not orbit a front/rear axle. |
| [`federation-council`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/federation-council/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/federation-council/material-review.png) | Hover ring; 1.20 m | Two arms; tri-blade terminal/tool and articulated open palm; torso pipes. | Rigid ring, central post and head shells; articulate at actual waist/neck bearings. Tool hub and each deployable blade need distinct rigid pivots if they unfold; otherwise keep the entire tool rigid. Pipes need dedicated flexible links. | Idle/heartbeat preserve a visible hover gap; walk leans/translates without wheel motion. Talk moves palm while ring stays circular; tool-use must not smear three blades into fingers. |
| [`knowledge-commons`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/knowledge-commons/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/knowledge-commons/material-review.png) | Four wheels in side housings/rockers; 1.20 m | Two arms; projector/light barrel, three-prong claw and roof lantern. | Wheel tires must be separated from unusually broad rocker/fender shells. Fit wheel centres from all elevations after restoring round hubs; the side render is heavily merged. Lantern stays on rigid spherical upper shell; projector barrel attaches rigidly to wrist. | Review sweeps projector while lantern and housing shapes stay fixed. Walk rotates tires only; do not rotate the whole side casing. Three claw prongs need their own segments if opening is required. |
| [`memory-stewardship`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/memory-stewardship/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/memory-stewardship/material-review.png) | Two continuous tracks; 1.20 m | Two arms; carried archive case and small crystal in claw; embedded torso crystal. | Rigid case/handle assigned to carrier hand, small crystal to a separate held-prop node, embedded crystal to torso. Isolate track belts/sprockets. Repair fused crystal/finger topology and maintain translucent surface boundaries. | Carry keeps case level; handoff moves held crystal without pulling torso crystal or case. Head turn must not move rear storage. No generic finger/torso smoothing across crystal contacts. |
| [`operations-release`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/operations-release/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/operations-release/material-review.png) | Four-wheel chassis; 1.20 m | Two arms; signal baton and tray carrying parcels; cylindrical torso. | Four independently centred wheels, rigid fenders. Baton shaft/guard and tray are hand props; parcels are separate tray children for stable transport. Keep cylindrical shell rigid at waist and head bearing. | Carry maintains a level tray and seated parcels; done can lift baton without sweeping parcels through torso. Walk wheel hubs must not include tray fingers at similar height. |
| [`pr-merge-train`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/pr-merge-train/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/pr-merge-train/material-review.png) | Two continuous tracks; 1.20 m | Two arms; large clamp and box inspection tool; low coupling/stabilizer machinery at front/rear. | Separate clamp jaws, rigid wrist housing and box tool. Fit dedicated coupler extension/hinge/foot chains; these low appendages are not track vertices. Preserve rear mechanism length and inspect it from both sides. | Handoff/carry extend clamp without moving coupler. Work may deploy coupler supports; walk stows them and drives tracks. Generated clamp and coupler edges need restoration before hinge tests. |
| [`research-lab`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/research-lab/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/research-lab/material-review.png) | Four roller/mecanum-style wheels; 1.20 m | Two main arms plus a distinct articulated shoulder camera boom; held vessel and probe/claw. | Add camera_boom base/mid/head bones independently of neck and main arm. Four wheel hubs need their own measured axes; small outer rollers may be represented as a rigid wheel detail initially, with no claim of lateral mecanum simulation. Vessel frame/glass and probe stay rigid to hands. | Review aims camera while both hands remain fixed; work handles vessel without bending frame or pulling torso specimen. Camera boom must not rotate with head or be absorbed by generic shoulder weights. |
| [`research-review-board`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/research-review-board/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/research-review-board/material-review.png) | Three radial support/roller feet; 1.20 m | Two arms; magnifier and crystal clamp/instrument, head sensor and rear module. | Fit three radial leg/support chains and each terminal roller if present after cleanup. Card front/back/side foot projections vary; generated front/side both show the third support, so do not reduce it to two feet. Magnifier rim/glass and instrument housing need rigid attachments; clamp jaws separate from crystal. | Review alternates magnifier and sample inspection. Supports stay planted or roll correctly; no biped leg reuse. Keep third support clear during turning and magnifier rim circular during wrist movement. |
| [`upstream-hermes-maintenance`](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/upstream-hermes-maintenance/source-card.png)<br>[Generated render](../../../apps/desktop/public/lunar-city/worker-multiview-2026-09-07/upstream-hermes-maintenance/material-review.png) | Two continuous tracks; 1.20 m | One full clamp arm plus a short arm/flexible cable ending in a probe; tall antenna. | Generated mesh subsequently inspected: cable survives as a continuous curved tube, while the main clamp is heavily fused/rough in side view. Restore clamp geometry, inventory track hubs and isolate cable from shell. Use a segmented cable chain following its curve, with a rigid probe tip and shoulder connector; do not stretch one forearm bone along the entire cable. | Walk drives tracks with cable stowed clear; work bends cable along its length while probe remains straight. Clamp and antenna must remain independent of cable control. Require full geometry review before any fitting acceptance. |

## Required extra controls and explicit ambiguity

**Per-wheel pivots:** archive, content, community, core-runtime, editorial, knowledge, operations
and research-lab. Editorial needs six rolling hubs, while the others show a four-wheel layout;
core-runtime's central underbody projection still needs classification against the actual mesh.
The two tripod variants need measured terminal roller axes if those endpoints are to roll.
Tracked variants need separate sprocket/housing/belt treatment, not four wheel masks.

**Additional chains:** arts utility branches/palette linkage; research-lab shoulder camera boom;
core-runtime stabilizers and braced forearms; PR-merge coupler/support mechanism; upstream flexible
probe cable. Arts views are not a mechanically consistent blueprint: preserve the visible
branches, resolve their attachments against front/side/plan and the generated model, and approve
that topology before deciding an exact final bone count. A camera boom is not a spare humanoid arm.

**Rigid held equipment:** every scanner, tablet, camera/light frame, palette, brush/pen, hammer,
projector, archive case, crystal, tray/parcel, baton, magnifier and clamp housing in the table
needs a deliberate controlling attachment. Do not use the same finger skinning region for both
hand and held object. Independently opening jaws or fingers need independent segments.

## Export and game acceptance per worker

The runtime's existing plain clip names are:
`idle`, `walk`, `talk`, `listen`, `work`, `tool-use`, `carry`, `handoff`, `queue`, `wait`,
`blocked`, `failed`, `review`, `triage`, `heartbeat`, `rest`, `done`.
Keep these names even for rolling/hovering travel; their physical implementation differs.
These gestures are presentation only, not evidence that an operational task succeeded.

- [ ] Export only evaluated, real clips. A held state is legitimate and must be reported as held;
      a clip name, bone count or “zero unweighted” statistic is not movement acceptance.
- [ ] Keep horizontal root translation out of locomotion clips: the navigation wrapper owns it.
      Author and record travel speed/stride distance. Rolling rate follows `angular speed = travel
      speed / wheel radius`; stop rolling at rest. The current default movement speed is 1.2 m/s.
      A fixed one-revolution clip is not automatically synchronized with that speed or radius.
- [ ] For tracks, author real sprocket/belt motion or clearly retain a static review mechanism.
      Whole-pod rotation is invalid. For hover bases, record clearance and keep the ring rigid.
      Tripod rolling versus stepping must be explicitly resolved before authoring travel.
- [ ] Inspect work/carry/review/tool-use separately: move each independent tool/auxiliary chain
      while the others remain fixed. Verify prop shape, grip contact, hose length and clearance.
- [ ] Sample every keyed frame and in-between extrema after GLB reimport, including non-quarter
      times. Baseline heartbeat was invisible at quarter-cycle samples because it uses two sine
      cycles; sampling frame 5 captured its real movement. Check both loop seams and finite endings.
- [ ] Measure edge-length preservation on rigid parts, finite positions and grounded contact at
      every sampled pose. A “no edge over 3x” guard detects catastrophic failures but is far too
      permissive to approve rigid metal; look for near-zero relative shape change within each part.
- [ ] Render front, both sides, rear and an underside view at rest and difficult poses. Check hands
      against held props, feet/wheels against ground, wheel hubs against fenders and supports against
      body. Geometry missing in the card-to-mesh result must be repaired before deformation grading.
- [ ] Reimport in a fresh Blender scene, then use the actual Babylon skin check with independent
      clones. Confirm one clip per character, actual normal idle, explicit reduced-motion parking,
      no cross-talk and near/mid/near transitions. Baseline's reviewed runtime path demonstrates this
      integration mechanism, not specialist acceptance.
- [ ] Preserve exact `kitId` and worker identity, then supply separately reviewed LODs before fleet
      promotion. Current opt-in baseline affects only neutral near workers; specialist kits remain
      historical. Do not replace every aggregate with a high-detail skinned model.

## Suggested fitting order

1. Repair core-runtime's wheel/stabilizer segmentation and editorial's six hubs; these expose the
   strongest generic lower-body assumptions. Validate one wheel/part at a time through 360 degrees.
2. Repair arts' auxiliary-branch topology and research-lab's camera boom before adding animation.
3. Fit biped hard-surface joints/props; then four-wheel characters using measured individual hubs.
4. Fit tracked, tripod and hover bases with their distinct travel semantics; finish cable/prop chains.
5. Run each character's complete per-clip and LOD checks before adding it to the optional runtime pack.

This order is a fitting recommendation. It does not supersede the Blender task's saved rejection
or acceptance decisions, and it does not authorize promotion of any current generic export.
