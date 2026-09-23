import { scalarClip, stateClips } from './animation.mjs'
import { beamBetween, box, capsule, cone, cylinder, group, sphere, torus } from './primitives.mjs'

export function addConsoleBank(
  scene,
  name,
  parent,
  { accent = 'signal-emissive', count = 3, position = [0, 1, 0], width = 1 } = {}
) {
  const consoles = group(scene, name, parent, { position })
  for (let index = 0; index < count; index += 1) {
    const x = (index - (count - 1) / 2) * width
    box(scene, `${name}:desk:${index}`, {
      depth: 0.65,
      height: 0.54,
      material: 'charcoal-structure',
      parent: consoles,
      position: [x, 0, 0],
      width: width * 0.82
    })
    box(scene, `${name}:screen:${index}`, {
      depth: 0.1,
      height: 0.44,
      material: accent,
      parent: consoles,
      position: [x, 0.52, -0.26],
      rotation: [-0.2, 0, 0],
      width: width * 0.62
    })
  }
  return consoles
}

export function addWorkbenches(
  scene,
  name,
  parent,
  { accent = 'signal-emissive', count = 2, position = [0, 0.7, 0] } = {}
) {
  const benches = group(scene, name, parent, { position })
  for (let index = 0; index < count; index += 1) {
    const x = (index - (count - 1) / 2) * 2.25
    box(scene, `${name}:top:${index}`, {
      depth: 1.1,
      height: 0.18,
      material: 'bone-metal',
      parent: benches,
      position: [x, 0.55, 0],
      width: 1.8
    })
    box(scene, `${name}:base:${index}`, {
      depth: 0.72,
      height: 0.72,
      material: 'charcoal-structure',
      parent: benches,
      position: [x, 0.12, 0],
      width: 1.15
    })
    cylinder(scene, `${name}:fixture:${index}`, {
      diameter: 0.26,
      height: 0.52,
      material: accent,
      parent: benches,
      position: [x + 0.5, 0.92, 0]
    })
  }
  return benches
}

export function addTelescope(scene, parent) {
  const telescope = group(scene, 'research-lab:telescope', parent, {
    position: [1.8, 2.8, 0.3],
    rotation: [0, 0, -0.2]
  })
  cylinder(scene, 'research-lab:telescope:tube', {
    diameter: 1.25,
    height: 3.6,
    material: 'bone-metal',
    parent: telescope,
    rotation: [0, 0, Math.PI / 2]
  })
  cylinder(scene, 'research-lab:telescope:lens', {
    diameter: 1.38,
    height: 0.22,
    material: 'signal-emissive',
    parent: telescope,
    position: [-1.78, 0, 0],
    rotation: [0, 0, Math.PI / 2]
  })
  cylinder(scene, 'research-lab:telescope:mount', {
    diameter: 0.48,
    height: 2.6,
    material: 'charcoal-structure',
    parent: telescope,
    position: [0, -1.55, 0]
  })
  scalarClip(scene, 'telescope-scan', telescope, 'rotation.y', 0.72, { duration: 72, start: -0.42 })
  return telescope
}

export function addPortal(scene, parent) {
  const portal = group(scene, 'review-office:portal', parent, {
    position: [-1.6, 2.5, -1.5]
  })
  const chamber = cylinder(scene, 'review-office:portal:chamber', {
    diameter: 2.5,
    height: 4.4,
    material: 'archive-emissive',
    parent: portal,
    tessellation: 14
  })
  chamber.metadata = { keepSeparate: true }
  for (const y of [-2.15, -0.9, 0.9, 2.15])
    torus(scene, `review-office:portal:ring:${y}`, {
      diameter: y === 0 ? 2.8 : 3,
      material: y % 2 ? 'archive-emissive' : 'bone-metal',
      parent: portal,
      position: [0, y, 0],
      tessellation: 14,
      thickness: 0.18
    })
  for (const side of [-1, 1])
    box(scene, `review-office:portal:pillar:${side}`, {
      depth: 0.5,
      height: 4.8,
      material: 'bone-metal',
      parent: portal,
      position: [side * 1.45, 0, 0],
      width: 0.5
    })
  scalarClip(scene, 'portal-idle', portal, 'rotation.y', Math.PI * 2, { duration: 120 })
  return portal
}

export function addPlants(scene, parent) {
  const plants = group(scene, 'garden:plants', parent)
  const patches = [
    [-4.8, 0, -2.7],
    [-3.5, 0, 2.7],
    [-1.8, 0, -3.8],
    [0.2, 0, 3.3],
    [2.3, 0, -3.1],
    [3.8, 0, 2.2],
    [5.1, 0, -0.4],
    [-0.5, 0, -0.8],
    [-5.5, 0, 0.1],
    [1.0, 0, 0.2],
    [-2.9, 0, 1.1],
    [3.0, 0, -1.0]
  ]
  for (const [index, [x, y, z]] of patches.entries()) {
    cylinder(scene, `garden:plant:stem:${index}`, {
      diameter: 0.18,
      height: 1.05 + (index % 3) * 0.22,
      material: 'garden-green',
      parent: plants,
      position: [x, y + 0.38, z]
    })
    cone(scene, `garden:plant:leaf-a:${index}`, {
      diameterBottom: 0.86,
      height: 1.18,
      material: 'garden-green',
      parent: plants,
      position: [x - 0.22, y + 1.0, z],
      rotation: [0, 0, -0.48]
    })
    cone(scene, `garden:plant:leaf-b:${index}`, {
      diameterBottom: 0.76,
      height: 1.06,
      material: 'garden-green',
      parent: plants,
      position: [x + 0.24, y + 0.92, z],
      rotation: [0, 0, 0.56]
    })
    if (index % 2 === 0)
      sphere(scene, `garden:plant:flower:${index}`, {
        diameter: 0.48,
        material: 'signal-emissive',
        parent: plants,
        position: [x, y + 1.55, z],
        segments: 6
      })
  }
  return plants
}

function addSignGlyph(scene, name, parent, accent, glyph) {
  const glyphRoot = group(scene, `${name}:identity-glyph`, parent, { position: [-1.1, 0, 0.15] })
  const stroke = (suffix, position, width, height, rotation = [0, 0, 0]) =>
    box(scene, `${name}:identity-glyph:${suffix}`, {
      depth: 0.06,
      height,
      material: accent,
      parent: glyphRoot,
      position,
      rotation,
      width
    })

  if (glyph === 'book') {
    stroke('left-cover', [-0.26, 0, 0], 0.48, 0.5, [0, 0, 0.12])
    stroke('right-cover', [0.26, 0, 0], 0.48, 0.5, [0, 0, -0.12])
    stroke('spine', [0, -0.02, -0.02], 0.08, 0.58)
  } else if (glyph === 'flask') {
    stroke('neck', [0, 0.17, 0], 0.16, 0.34)
    stroke('bowl', [0, -0.14, 0], 0.62, 0.34)
    stroke('liquid', [0, -0.2, -0.04], 0.46, 0.08)
  } else if (glyph === 'crate') {
    stroke('body', [0, 0, 0], 0.64, 0.56)
    stroke('slash-a', [0, 0, -0.04], 0.09, 0.72, [0, 0, 0.75])
    stroke('slash-b', [0, 0, -0.05], 0.09, 0.72, [0, 0, -0.75])
  } else if (glyph === 'review') {
    torus(scene, `${name}:identity-glyph:ring`, {
      diameter: 0.65,
      material: accent,
      parent: glyphRoot,
      rotation: [Math.PI / 2, 0, 0],
      tessellation: 10,
      thickness: 0.1
    })
    stroke('check', [0.08, -0.04, -0.06], 0.1, 0.58, [0, 0, -0.7])
  } else if (glyph === 'council') {
    stroke('dais', [0, -0.2, 0], 0.8, 0.12)
    for (const [index, x] of [-0.28, 0, 0.28].entries()) stroke(`seat:${index}`, [x, 0.08, 0], 0.16, 0.45)
  }
  return glyphRoot
}

const SIGN_FONT = Object.freeze({
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
  B: ['11110', '10001', '10001', '11110', '10001', '10001', '11110'],
  C: ['01111', '10000', '10000', '10000', '10000', '10000', '01111'],
  D: ['11110', '10001', '10001', '10001', '10001', '10001', '11110'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  G: ['01111', '10000', '10000', '10111', '10001', '10001', '01111'],
  H: ['10001', '10001', '10001', '11111', '10001', '10001', '10001'],
  I: ['11111', '00100', '00100', '00100', '00100', '00100', '11111'],
  L: ['10000', '10000', '10000', '10000', '10000', '10000', '11111'],
  N: ['10001', '11001', '10101', '10011', '10001', '10001', '10001'],
  O: ['01110', '10001', '10001', '10001', '10001', '10001', '01110'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  S: ['01111', '10000', '10000', '01110', '00001', '00001', '11110'],
  T: ['11111', '00100', '00100', '00100', '00100', '00100', '00100'],
  U: ['10001', '10001', '10001', '10001', '10001', '10001', '01110'],
  V: ['10001', '10001', '10001', '10001', '10001', '01010', '00100'],
  W: ['10001', '10001', '10001', '10101', '10101', '11011', '10001'],
  Y: ['10001', '10001', '01010', '00100', '00100', '00100', '00100']
})

function addSignText(scene, name, parent, accent, text, width, centerX) {
  const normalized = text.toUpperCase().slice(0, 10)
  const cell = Math.min(0.105, width / Math.max(1, normalized.length * 6.4))
  const glyphWidth = cell * 5
  const spacing = cell * 1.35
  const totalWidth = normalized.length * (glyphWidth + spacing) - spacing
  const originX = centerX - totalWidth / 2
  const originY = -0.27
  for (const [charIndex, character] of [...normalized].entries()) {
    const rows = SIGN_FONT[character]
    if (!rows) continue
    const x = originX + charIndex * (glyphWidth + spacing)
    for (const [rowIndex, row] of rows.entries()) {
      let runStart = -1
      for (let column = 0; column <= row.length; column += 1) {
        const filled = column < row.length && row[column] === '1'
        if (filled && runStart < 0) runStart = column
        if ((!filled || column === row.length) && runStart >= 0) {
          const runWidth = column - runStart
          box(scene, `${name}:text:${charIndex}:${rowIndex}:${runStart}`, {
            depth: 0.045,
            height: cell,
            material: accent,
            parent,
            position: [x + (runStart + runWidth / 2) * cell, originY + (6 - rowIndex) * cell, 0.145],
            width: runWidth * cell
          })
          runStart = -1
        }
      }
    }
  }
}

export function addSign(
  scene,
  name,
  parent,
  { accent, glyph = null, label = null, position = [0, 0, 0], rotation = [0, 0, 0], width = 3.4 } = {}
) {
  const sign = group(scene, name, parent, { position, rotation })
  box(scene, `${name}:panel`, { depth: 0.22, height: 0.86, material: 'charcoal-structure', parent: sign, width })
  if (label) {
    addSignText(scene, name, sign, accent, label, width * (glyph ? 0.56 : 0.82), glyph ? width * 0.16 : 0)
  } else {
    box(scene, `${name}:glow`, {
      depth: 0.05,
      height: 0.16,
      material: accent,
      parent: sign,
      position: [glyph ? 0.62 : 0, 0.14, 0.13],
      width: width * (glyph ? 0.52 : 0.74)
    })
    box(scene, `${name}:glyph`, {
      depth: 0.05,
      height: 0.14,
      material: accent,
      parent: sign,
      position: [glyph ? 0.62 : 0, -0.18, 0.13],
      width: width * (glyph ? 0.32 : 0.42)
    })
  }
  if (glyph) addSignGlyph(scene, name, sign, accent, glyph)
  return sign
}

export function buildBus(scene) {
  const root = group(scene, 'bus:root')
  const near = group(scene, 'bus:lod:near', root)
  const far = group(scene, 'bus:lod:far', root)
  const cabin = group(scene, 'bus:cabin', near)
  box(scene, 'bus:cabin:lower', {
    depth: 2.65,
    height: 1.35,
    material: 'lunar-rust',
    parent: cabin,
    position: [0, 1.22, 0],
    width: 6.8
  })
  box(scene, 'bus:cabin:upper', {
    depth: 2.42,
    height: 1.05,
    material: 'bone-metal',
    parent: cabin,
    position: [-0.2, 2.25, 0],
    width: 5.55
  })
  box(scene, 'bus:cabin:windshield', {
    depth: 2.46,
    height: 0.62,
    material: 'signal-emissive',
    parent: cabin,
    position: [2.72, 2.26, 0],
    rotation: [0, 0, -0.12],
    width: 0.5
  })
  const wheels = group(scene, 'bus:wheels', near)
  for (const x of [-2.15, 2.05]) {
    for (const z of [-1.32, 1.32])
      cylinder(scene, `bus:wheel:${x}:${z}`, {
        diameter: 0.92,
        height: 0.28,
        material: 'bone-metal',
        parent: wheels,
        position: [x, 0.62, z],
        rotation: [Math.PI / 2, 0, 0]
      })
  }
  const signal = group(scene, 'bus:signal', near)
  box(scene, 'bus:signal:side-a', {
    depth: 0.07,
    height: 0.32,
    material: 'signal-emissive',
    parent: signal,
    position: [-0.7, 1.7, -1.37],
    width: 2.2
  })
  box(scene, 'bus:signal:side-b', {
    depth: 0.07,
    height: 0.32,
    material: 'signal-emissive',
    parent: signal,
    position: [-0.7, 1.7, 1.37],
    width: 2.2
  })
  group(scene, 'bus:camera', root, { position: [0, 5, 9] })

  box(scene, 'bus:far:body', {
    depth: 2.5,
    height: 2.1,
    material: 'lunar-rust',
    parent: far,
    position: [0, 1.45, 0],
    width: 6.4
  })
  stateClips(scene, near, ['idle', 'arrive', 'depart'])
  return root
}

export function buildTriage(scene) {
  const root = group(scene, 'triage:root')
  const near = group(scene, 'triage:lod:near', root)
  const far = group(scene, 'triage:lod:far', root)
  const interior = group(scene, 'triage:open-interior', near)
  box(scene, 'triage:floor', {
    depth: 7.4,
    height: 0.5,
    material: 'charcoal-structure',
    parent: near,
    position: [0, 0.25, 0],
    width: 9.5
  })
  box(scene, 'triage:back-wall', {
    depth: 0.55,
    height: 4.4,
    material: 'charcoal-structure',
    parent: interior,
    position: [0, 2.45, -3.05],
    width: 7.6
  })
  for (const side of [-1, 1]) {
    box(scene, `triage:side-wall:${side}`, {
      depth: 5.9,
      height: 4.15,
      material: side < 0 ? 'triage-amber' : 'bone-metal',
      parent: interior,
      position: [side * 3.7, 2.32, -0.25],
      width: 0.72
    })
    for (let rib = 0; rib < 3; rib += 1)
      box(scene, `triage:side-rib:${side}:${rib}`, {
        depth: 0.42,
        height: 3.2 - rib * 0.35,
        material: rib === 1 ? 'triage-amber' : 'charcoal-structure',
        parent: interior,
        position: [side * 4.02, 2.05, -2 + rib * 1.8],
        rotation: [0, 0, side * 0.08],
        width: 0.4
      })
  }
  box(scene, 'triage:canopy', {
    depth: 6.4,
    height: 0.72,
    material: 'triage-amber',
    parent: near,
    position: [0, 4.92, -0.18],
    width: 8.7
  })
  box(scene, 'triage:front-header', {
    depth: 0.62,
    height: 0.82,
    material: 'triage-amber',
    parent: near,
    position: [0, 4.25, 2.82],
    width: 8.1
  })
  for (const side of [-1, 1])
    cylinder(scene, `triage:front-post:${side}`, {
      diameter: 0.7,
      height: 4.1,
      material: 'bone-metal',
      parent: near,
      position: [side * 3.65, 2.2, 2.72],
      tessellation: 8
    })
  const door = group(scene, 'triage:door', near, { position: [3.32, 1.75, -0.2], rotation: [0, Math.PI / 2, 0] })
  box(scene, 'triage:door:panel', {
    depth: 0.22,
    height: 2.7,
    material: 'charcoal-structure',
    parent: door,
    width: 1.35
  })
  const cross = group(scene, 'triage:cross', near, { position: [-2.25, 4.28, 3.16] })
  box(scene, 'triage:cross:h', { depth: 0.12, height: 0.34, material: 'bone-metal', parent: cross, width: 1.45 })
  box(scene, 'triage:cross:v', { depth: 0.12, height: 1.45, material: 'bone-metal', parent: cross, width: 0.34 })
  const station = group(scene, 'triage:station', interior)
  addConsoleBank(scene, 'triage:station:consoles', station, {
    accent: 'triage-amber',
    count: 2,
    position: [1.55, 1.05, -2.55],
    width: 1.25
  })
  const bed = group(scene, 'triage:medical-bed', interior, { position: [-1.35, 1, -0.3] })
  box(scene, 'triage:medical-bed:base', {
    depth: 3.7,
    height: 0.65,
    material: 'bone-metal',
    parent: bed,
    rotation: [0, -0.14, 0],
    width: 1.65
  })
  box(scene, 'triage:medical-bed:pad', {
    depth: 3.35,
    height: 0.28,
    material: 'triage-amber',
    parent: bed,
    position: [0, 0.48, 0],
    rotation: [0, -0.14, 0],
    width: 1.45
  })
  const supplies = group(scene, 'triage:supply-bank', interior)
  for (let index = 0; index < 8; index += 1) {
    cylinder(scene, `triage:supply:${index}`, {
      diameter: 0.52 + (index % 2) * 0.12,
      height: 1.45 + (index % 3) * 0.24,
      material: index % 3 ? 'bone-metal' : 'triage-amber',
      parent: supplies,
      position: [-2.8 + (index % 4) * 0.75, 1.15 + Math.floor(index / 4) * 1.55, -2.65],
      tessellation: 8
    })
  }
  for (let panel = 0; panel < 6; panel += 1)
    box(scene, `triage:status-panel:${panel}`, {
      depth: 0.12,
      height: 0.5 + (panel % 2) * 0.25,
      material: panel % 2 ? 'triage-amber' : 'bone-metal',
      parent: interior,
      position: [-2.5 + panel, 3.15 + (panel % 2) * 0.38, -2.72],
      width: 0.68
    })
  for (let step = 0; step < 3; step += 1)
    box(scene, `triage:front-step:${step}`, {
      depth: 0.7 + step * 0.42,
      height: 0.2,
      material: step === 1 ? 'triage-amber' : 'charcoal-structure',
      parent: near,
      position: [0, 0.12 + step * 0.14, 3.2 + step * 0.2],
      width: 5.9 - step * 0.45
    })
  group(scene, 'triage:camera', root, { position: [0, 6, 10] })
  box(scene, 'triage:far:shell', {
    depth: 5.7,
    height: 4.5,
    material: 'triage-amber',
    parent: far,
    position: [0, 2.55, -0.5],
    width: 7.4
  })
  box(scene, 'triage:far:cross', {
    depth: 0.1,
    height: 0.44,
    material: 'bone-metal',
    parent: far,
    position: [0, 4.15, 2.42],
    width: 1.4
  })
  scalarClip(scene, 'door-open', door, 'rotation.y', -1.2, { duration: 36 })
  scalarClip(scene, 'lights-idle', cross, 'rotation.y', 0.08, { duration: 42 })
  scalarClip(scene, 'triage-station-idle', station, 'rotation.y', 0.055, { duration: 64 })
  return root
}

export function addAntler(scene, name, parent, side) {
  const antler = group(scene, name, parent, { position: [side * 0.44, 1.2, 0] })
  beamBetween(scene, `${name}:stem`, [0, 0, 0], [side * 0.32, 1.1, 0], {
    height: 0.12,
    material: 'bone-metal',
    parent: antler,
    width: 0.14
  })
  for (let index = 0; index < 3; index += 1) {
    beamBetween(
      scene,
      `${name}:tine:${index}`,
      [side * 0.1, 0.34 + index * 0.25, 0],
      [side * (0.48 + index * 0.08), 0.67 + index * 0.31, 0],
      { height: 0.1, material: 'bone-metal', parent: antler, width: 0.1 }
    )
  }
  return antler
}

export function addRobotTool(scene, name, parent, kind, position) {
  const tool = group(scene, name, parent, { position })
  if (kind === 'ring')
    torus(scene, `${name}:shape`, {
      diameter: 0.58,
      material: 'signal-emissive',
      parent: tool,
      tessellation: 8,
      thickness: 0.11
    })
  else if (kind === 'satchel')
    box(scene, `${name}:shape`, {
      depth: 0.18,
      height: 0.46,
      material: 'charcoal-structure',
      parent: tool,
      width: 0.56
    })
  else if (kind === 'antenna') {
    cylinder(scene, `${name}:stem`, { diameter: 0.09, height: 0.58, material: 'charcoal-structure', parent: tool })
    sphere(scene, `${name}:tip`, {
      diameter: 0.22,
      material: 'signal-emissive',
      parent: tool,
      position: [0, 0.35, 0],
      segments: 6
    })
  } else {
    box(scene, `${name}:handle`, {
      depth: 0.12,
      height: 0.62,
      material: 'charcoal-structure',
      parent: tool,
      width: 0.12
    })
    box(scene, `${name}:head`, {
      depth: 0.18,
      height: 0.2,
      material: 'charcoal-structure',
      parent: tool,
      position: [0, 0.34, 0],
      width: 0.48
    })
  }
  return tool
}

export function addRobotLimb(scene, name, parent, position, rotation = [0, 0, 0]) {
  const limb = group(scene, name, parent, { position, rotation })
  capsule(scene, `${name}:shell`, { height: 0.7, material: 'bone-metal', parent: limb, radius: 0.11, tessellation: 6 })
  sphere(scene, `${name}:joint`, {
    diameter: 0.24,
    material: 'charcoal-structure',
    parent: limb,
    position: [0, -0.34, 0],
    segments: 6
  })
  return limb
}

export function addAnimalTail(
  scene,
  name,
  parent,
  { material = 'lunar-rust', position = [0, 0, 0], scale = [1, 1, 1] } = {}
) {
  const tail = group(scene, name, parent, { position, rotation: [0.7, 0, -0.8], scale })
  cone(scene, `${name}:shape`, {
    diameterBottom: 0.65,
    diameterTop: 0.18,
    height: 1.8,
    material,
    parent: tail,
    tessellation: 6
  })
  return tail
}
