# Monkey hand and coat separation audit

The monkey stays on static material review. Repositioning its arm hinge cannot fix coat faces
assigned to the palm. The next fitting step needs an explicit semantic surface split.

Read-only evidence: preserved `monkey-poet/source-card.png`, `material-review-v4.png`,
`corrected-right-v4.png`, and the saved `rig-stress-pose.png`, under
`apps/desktop/public/lunar-city/multiview-2026-09-07/`. The stress image can be replaced by later
Blender experiments; these observations describe the image inspected during this audit.
The immutable material GLB hash and structural measurements are in
[evidence/monkey-hand-coat-topology.json](evidence/monkey-hand-coat-topology.json).

The front card shows bare hands outside the coat skirt, with broad sleeve cuffs above the
palms. The side card overlaps the hand and skirt in projection; that overlap does not specify
a welded surface or establish their hidden depth. The generated front looks plausible at rest,
but the side has a visually continuous noisy cuff/hand/skirt region. In the inspected stress
render, the raised hand on image-right pulls the right skirt panel into a broad triangular
flap. The cuff/hem stretches as a band rather than just individual fingers. This locates the
failure at the palm/cuff and lateral skirt boundary, not solely at the shoulder.

The v4 material GLB contains one mesh, no skin, 59,995 triangles and a 1.64924 m local height.
Welding duplicate UV-seam positions to 1 micrometre for analysis gives a largest component of
58,838 triangles (98.1%). This proves that loose-part extraction alone cannot recover the
intended anatomy. It does not prove a particular hand/coat contact loop: the normal body,
sleeve and coat are also legitimately connected. No defensible automatic cut loop was found
in this audit, and no source geometry or weights were changed.

## Concrete repair sequence

1. Preserve v4 and the rejected rig. In a working copy, paint separate face sets for bare
   palm/digits, sleeve/cuff, torso, left/right coat skirt, legs, belt/books and tail. Inspect
   orthographic front, side and rear with hidden faces visible. Use card landmarks and actual
   surface continuity, not another height/X range. Label ambiguous occluded faces explicitly.
2. Inspect the transition where each bare hand approaches the lateral skirt and sleeve cuff.
   Keep the true wrist/cuff opening. Where reconstruction has bridged hand to skirt, remove
   only the labelled bridge patch and duplicate the contact boundary vertices. Rebuild the
   hidden palm side and coat underside as separate surfaces with a small visible clearance.
   Do not cut the whole coat at the hand's height or delete all connected low-side geometry.
3. Retopologize poorly resolved fingers and palm locally, preserving their card silhouette.
   Use wrist and knuckle loops with enough spacing to bend without turning triangles into
   long strips. Rebuild the cuff lip as a stable sleeve boundary. Close unintended holes and
   reproject the baked colour/normal onto the repaired mesh; inspect the newly exposed sides.
4. Weight sleeve to upper/forearm with a real elbow transition; palm/digits only to wrist and
   finger bones. Give skirt panels pelvis/spine or dedicated skirt controls; books and belt
   stay rigid on pelvis/torso. Keep legs on leg chains and tail on its own chain. Prohibit
   adjacency weight smoothing across the semantic hand/skirt and prop/body boundaries.
5. Test each hand independently: arm raise, elbow bend, wrist twist, finger curl, then walking
   and all six chat states. Compare the labelled skirt vertices against the same pose with
   only the hand control moving: wrist/finger motion must not move skirt vertices. Check
   silhouette and self-intersection from front, both sides and rear, including in-between
   keys. A maximum edge-stretch threshold alone does not establish acceptable deformation.
6. Export a versioned review only after these checks. Reimport, verify metre height, skin and
   individual clip targets, then repeat actual skin-motion and independent-clone tests in
   Babylon. Keep the static v4 runtime selection until the repaired export passes visual QA.

The key implementation artifact is the face-set/part assignment, saved with the Blender
model. Without it, another spatial mask will reproduce the same ambiguity when the hand,
coat and leg occupy nearby coordinates.
