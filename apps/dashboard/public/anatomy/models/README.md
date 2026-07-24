# Adding a high-detail anatomy model (`body.glb`)

Drop a glTF-binary file named **`body.glb`** in this folder
(`apps/dashboard/public/anatomy/models/body.glb`) and the Anatomy Explorer's
**"Load high-detail model"** button will use it on WebGL devices. Nothing else
to configure — the loader **auto-centres and auto-scales** whatever you supply,
so you don't need to fuss over units or position in Blender.

The file is intentionally **not** committed (licensing + size). Here's how to
make one from **Z-Anatomy** (CC-BY-SA).

## Step by step (Blender)

1. **Get Blender** (free): https://www.blender.org/download/ — 3.6 or newer.
2. **Get Z-Anatomy**: https://github.com/LluisV/Z-Anatomy (or the "Z-Anatomy"
   releases). Download the `.blend` atlas file(s).
3. **Open** the `.blend` in Blender. You'll see thousands of precisely-named
   Latin structures organised into collections (Skeleton, Muscles, Viscera…).
4. **Keep what you need.** For a mobile-friendly file, delete collections you
   don't want (nerves, lymphatics, fine vessels). Major systems are plenty.
5. **Group into our structures (important for highlighting).** For each of our
   structures, select its object(s) and **Join** them (`Ctrl+J`), then rename
   the joined object (double-click in the Outliner) to the matching **id**:

   `skin, musculature, skull, spine, ribcage, pelvis, arm_bones, leg_bones,`
   `brain, heart, lungs, liver, stomach, intestines, kidneys, bladder, spleen,`
   `pancreas, trachea, thyroid, gallbladder, diaphragm`

   > Meshes you don't rename still render — they just land in the **Organs**
   > layer and aren't individually click-highlightable. So rename the handful
   > you care about and leave the rest; it still looks great.
6. **Decimate for the web.** Select the heavy meshes → Modifier ▸ **Decimate**
   (Collapse ~0.3–0.5) → Apply. Aim for a few MB total so phones stay smooth.
7. **Export.** `File ▸ Export ▸ glTF 2.0 (.glb)`. In the export panel:
   - Format: **glTF Binary (.glb)**
   - Include: **Selected Objects** (if you only want your kept set)
   - Compression: enable **Draco mesh compression** (big size win)
   - Save it here as **`body.glb`**.
8. **Reload the dashboard**, open **Health ▸ Anatomy Explorer**, ensure Quality
   is 3D/Auto, and click **Load high-detail model**. Done.

## Alternatives
- **BodyParts3D** (CC-BY-SA, life-science DB) — per-structure meshes already
  keyed to anatomy ids; convert OBJ→GLB and name per the list above.
- A ready-made **CC-BY** anatomy `.glb` from Sketchfab — same drop-in; rename
  meshes to our ids for targeting, keep the attribution.

## Attribution
Z-Anatomy and BodyParts3D are **CC-BY-SA** — keep their attribution wherever you
share screenshots or a deployed instance.
