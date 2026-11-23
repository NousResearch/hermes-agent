import * as React from "react";
import { Dots, FaceText, Line, Solid } from "../../kit/react/draw";
import { BASE_ART, CARRIAGE_ART, DIAL_ART, FINGER_ART, FRAME_ART, PLUNGER_ART, PROBE_ART, RAIL_ART, REST_OUTLINE, RIDER, RODS } from "./geometry";
import { StretchLive } from "./live";

const BASE = (
  <g key="base">
    <path className="iso-halo" d={BASE_ART.halo} />
    {BASE_ART.feet.map((parts, index) => (
      <g key={index}>
        {parts.map((paths, part) => (
          <Solid key={part} paths={paths} tone={part ? "mid" : "lo"} crease="faint" />
        ))}
      </g>
    ))}
    <Solid paths={BASE_ART.slab} tone="mid" crease="faint" />
    <Line d={BASE_ART.groove} tone="lo" />
    <Line d={BASE_ART.screwRings} tone="lo" />
    <Dots points={BASE_ART.screws} size={0.5} tone="mid" />
    <Solid paths={BASE_ART.plate} tone="lo" crease="none" />
    <Line d={BASE_ART.plateLine} tone="faint" />
    <Line d={BASE_ART.plateRule} tone="lo" />
    <Dots points={BASE_ART.plateScrews} size={0.45} tone="mid" />
  </g>
);

const RAIL = (
  <g key="rail">
    {RAIL_ART.feet.map((paths, index) => (
      <Solid key={index} paths={paths} tone="lo" crease="faint" />
    ))}
    <Dots points={RAIL_ART.footScrews} size={0.45} tone="mid" />
    {RAIL_ART.posts.map((paths, index) => (
      <Solid key={index} paths={paths} tone="mid" crease="faint" />
    ))}
    <Line d={RAIL_ART.postLines} tone="faint" />
    <Solid paths={RAIL_ART.rail} tone="mid" crease="faint" />
    <Line d={RAIL_ART.ruleBase} tone="faint" />
    <Line d={RAIL_ART.ruleMinor} tone="lo" />
    <Line d={RAIL_ART.ruleMajor} tone="mid" />
    <Dots points={RAIL_ART.railScrews} size={0.5} tone="mid" />
    {RAIL_ART.stops.map((paths, index) => (
      <Solid key={index} paths={paths} tone="mid" crease="faint" />
    ))}
  </g>
);

const RIDER_ART = (
  <g key="rider-art">
    <Solid paths={RIDER.body} tone="hi" crease="faint" />
    <Line d={RIDER.tick} tone="hi" />
  </g>
);

const BLOCK = (
  <g key="block">
    <Solid paths={PROBE_ART.block} tone="mid" crease="faint" />
    <Line d={PROBE_ART.blockLine} tone="faint" />
    <Dots points={PROBE_ART.blockScrews} size={0.45} tone="mid" />
    <Solid paths={PROBE_ART.thumb} tone="mid" crease="faint" />
    <Line d={PROBE_ART.thumbKnurl} tone="lo" />
    <Solid paths={PROBE_ART.pointer} tone="mid" crease="none" />
    <Line d={PROBE_ART.index} tone="lit" />
  </g>
);

const FRAME_BACK = (
  <g key="frame-back">
      <Solid paths={FRAME_ART.back} tone="mid" crease="faint" />
      <Dots points={FRAME_ART.backScrews} size={0.5} tone="mid" />
      <Solid paths={FRAME_ART.left} tone="mid" crease="faint" />
      <Dots points={FRAME_ART.leftScrews} size={0.5} tone="mid" />
      <Solid paths={FRAME_ART.floor} tone="lo" crease="none" flat />
      <Line d={FRAME_ART.grid} tone="faint" />
      <Line d={FRAME_ART.leftBores} tone="lo" />
  </g>
);

const RODS_ART = (
  <g key="rods-art">
    {RODS.map((paths, index) => (
      <Solid key={index} paths={paths} tone="mid" crease="none" />
    ))}
  </g>
);

const CARRIAGE = (
  <g key="carriage">
    <path className="iso-halo ms-shadow" d={CARRIAGE_ART.halo} />
    <Solid paths={CARRIAGE_ART.plate} tone="mid" crease="faint" />
    <Line d={CARRIAGE_ART.seatInner} tone="faint" />
    <Line d={CARRIAGE_ART.seat} tone="lo" />
    <Dots points={CARRIAGE_ART.rivets} size={0.4} tone="lo" />
    {CARRIAGE_ART.bushes.map((paths, index) => (
      <Solid key={index} paths={paths} tone="mid" crease="faint" />
    ))}
    <Dots points={CARRIAGE_ART.bushScrews} size={0.45} tone="mid" />
    <Line d={CARRIAGE_ART.bushBores} tone="lo" />
    <Solid paths={CARRIAGE_ART.contact} tone="mid" crease="faint" />
  </g>
);

const PLUNGER = (
  <g key="plunger">
    <Solid paths={PLUNGER_ART.rod} tone="mid" crease="none" />
    <Solid paths={PLUNGER_ART.collar} tone="mid" crease="faint" />
  </g>
);

const REST = <Line key="rest" d={REST_OUTLINE} tone="mid" dotted />;

const FRAME_FRONT = (
  <g key="frame-front">
      <Solid paths={FRAME_ART.right} tone="mid" crease="faint" />
      <Dots points={FRAME_ART.rightScrews} size={0.5} tone="mid" />
      <Line d={FRAME_ART.rodEnds} tone="lo" />
      <Dots points={FRAME_ART.rodNuts} size={0.55} tone="mid" />
      <Line d={FRAME_ART.bore} tone="lo" />
      <Solid paths={FRAME_ART.front} tone="mid" crease="faint" />
      <Dots points={FRAME_ART.frontScrews} size={0.5} tone="mid" />
      <Line d={FRAME_ART.ticksMinor} tone="lo" />
      <Line d={FRAME_ART.ticksMajor} tone="mid" />
      <Line d={FRAME_ART.zero} tone="hi" />
  </g>
);

const DIAL = (
  <g key="dial">
    <Solid paths={DIAL_ART.stand} tone="lo" crease="faint" />
    <Dots points={DIAL_ART.standScrews} size={0.45} tone="mid" />
    <Solid paths={DIAL_ART.sleeve} tone="mid" crease="faint" />
    <Solid paths={DIAL_ART.clamp} tone="mid" crease="none" />
    <Solid paths={DIAL_ART.lug} tone="mid" crease="faint" />
    <Solid paths={DIAL_ART.puck} tone="mid" crease="faint" />
    <Line d={DIAL_ART.band} tone="faint" />
    <Line d={DIAL_ART.knurl} tone="faint" />
    <Line d={DIAL_ART.bezel} tone="lo" />
    <Line d={DIAL_ART.face} tone="faint" />
    <Line d={DIAL_ART.minor} tone="lo" />
    <Line d={DIAL_ART.major} tone="mid" />
    <Dots points={DIAL_ART.give} size={0.55} tone="hi" />
    <FaceText transform={DIAL_ART.label} size={2.6} tone="lo" anchor="middle">
      px
    </FaceText>
    <Solid paths={DIAL_ART.crown} tone="mid" crease="faint" />
    <Solid paths={DIAL_ART.crownCap} tone="mid" crease="faint" />
  </g>
);

const FINGER = (
  <g key="finger">
    <Solid paths={FINGER_ART.stem} tone="mid" crease="none" />
    <Line d={FINGER_ART.stemMarks} tone="lo" />
    <Solid paths={FINGER_ART.collar} tone="mid" crease="faint" />
    <Solid paths={FINGER_ART.pad} tone="hi" crease="faint" />
    <Line d={FINGER_ART.padRing} tone="lo" />
  </g>
);

const ARM = (
  <g key="arm">
    <Solid paths={PROBE_ART.collar} tone="mid" crease="faint" />
    <Solid paths={PROBE_ART.housing} tone="mid" crease="faint" />
    <Line d={PROBE_ART.housingSeam} tone="faint" />
    <Solid paths={PROBE_ART.lock} tone="mid" crease="none" />
    <Solid paths={PROBE_ART.lockHead} tone="mid" crease="faint" />
    <Solid paths={PROBE_ART.cap} tone="mid" crease="faint" />
    <Solid paths={PROBE_ART.arm} tone="mid" crease="faint" />
    <Line d={PROBE_ART.armLine} tone="faint" />
    <Dots points={PROBE_ART.armScrews} size={0.45} tone="mid" />
  </g>
);

const LABEL =
  "A test rig for a glass button. A 120 × 32 capsule of glass sits on a carriage that slides on two guide rods inside a frame, held at rest by four coil springs, two at each end. A finger probe hangs from a carriage on a rail behind it: it comes down to press the glass, which swells, then slides along the rail to pull, and the glass leans after it, giving less the further it goes. A dial gauge at the end of the frame reads the lean. Let go and the springs bring it back, once past rest.";

export function StretchArt() {
  return (
    <StretchLive
      label={LABEL}
      base={BASE}
      rail={RAIL}
      rider={RIDER_ART}
      block={BLOCK}
      frameBack={FRAME_BACK}
      rods={RODS_ART}
      carriage={CARRIAGE}
      plunger={PLUNGER}
      rest={REST}
      frameFront={FRAME_FRONT}
      dial={DIAL}
      finger={FINGER}
      arm={ARM}
    />
  );
}
