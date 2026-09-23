# Beaver waterworks river-port study

Read-only source-card/render/GLB inspection for the organic island revision. No model or layout was edited.

The card PLAN puts a water channel across the dam on the local front/back axis. Current source GLB front faces +Z. Both FRONT and REAR card panels illustrate falling water, so the card does not uniquely establish upstream/downstream direction; select the physical river direction coherently without treating the illustration as a hydraulic specification. The SIDE panel is an angled elevation rather than a true orthographic side.

Measured source bounds: X −10.947213 to10.949093 m, Z −10.912091 to10.911608 m, height14.000001 m. At the revised building design scale0.75, width is16.422230 m and depth16.367774 m. With the pre-island placement center [−30.276,2.2,31.32] and yaw2.373142 radians, local +Z maps to world [.695022,0,−.718988]. Approximate centerline ports at the mesh's outer channel bounds are:

- Front: [−24.588144,2.2,25.436011].
- Rear: [−35.963856,2.2,37.203989].

These are bounding-centerline anchors, not surveyed separate water meshes. Recompute from center/yaw after any placement revision. Route through these ±Z ports, not the ±X side walls/banks. Source PLAN suggests about5–6 m water width after scale, estimated from the illustrated channel-to-building width ratio.

For actual geometry in the two central outer channel regions (`abs X <3.5`, front `Z>8`, rear `Z<−8`), source Y ranges from0.006 to0.159 m. Front median is0.046264 m and rear median0.048723 m. With0.75 scale and ground2.2 m, those medians place the existing flat channel surface around2.235–2.237 m. This measures vertices, not semantic water classification: the current material review incorrectly renders these regions as gray stone. A continuous river surface needs explicit material/bed separation and a deliberate vertical join rather than hiding the defect with an arbitrary world-height shift.

Evidence: [source card](../../../apps/desktop/public/lunar-city/building-multiview-2026-09-07/beaver/source-card.png), [front](../../../apps/desktop/public/lunar-city/building-multiview-2026-09-07/beaver/building-front.png), [side](../../../apps/desktop/public/lunar-city/building-multiview-2026-09-07/beaver/building-side.png), [top](../../../apps/desktop/public/lunar-city/building-multiview-2026-09-07/beaver/building-top.png), [source GLB](../../../apps/desktop/public/lunar-city/building-multiview-2026-09-07/beaver/building-review.glb). Hash captured in [static building receipt](evidence/building-static-quality-audit.json).
