import * as React from "react";
import { Dots, FaceText, Line, Solid } from "../../kit/react/draw";
import { BASE_ART, BOTTLE_ART, DECK_ART, LABEL, TRAY_ART } from "./geometry";
import { LiquidLive } from "./live";

function Lit({ id, children }: { id: string; children: React.ReactNode }) {
  return (
    <g data-part={id}>{children}</g>
  );
}

const BACK = (
  <g key="back">
    <Lit id="frame">
      <path className="iso-halo" d={BASE_ART.halo} />
      {BASE_ART.feet.map((parts, index) => (
        <g key={index}>
          {parts.map((paths, piece) => (
            <Solid key={piece} paths={paths} tone={piece ? "mid" : "lo"} crease="faint" />
          ))}
        </g>
      ))}
      <Solid paths={BASE_ART.slab} tone="mid" crease="faint" />
      <Line d={BASE_ART.groove} tone="lo" />
      <Line d={BASE_ART.inlay} tone="faint" />
      <Line d={BASE_ART.screwRings} tone="lo" />
      <Dots points={BASE_ART.screws} size={0.5} tone="mid" />
      <Solid className="ml-plate" paths={TRAY_ART.slab} tone="hi" crease="lo" />
      <Line d={TRAY_ART.groove} tone="lo" />
      <Line d={TRAY_ART.inlay} tone="faint" />
      {TRAY_ART.wells.map((well, index) => (
        <g key={index}>
          <path className="ml-floor" d={well.opening} />
          {well.wall ? <path className="ml-wall" d={well.wall} /> : null}
          <Line d={well.foot} tone="faint" />
          <Line d={well.opening} tone="mid" />
        </g>
      ))}
      <Dots points={TRAY_ART.counts} size={0.55} tone="mid" />
      <Line d={TRAY_ART.screwRings} tone="lo" />
      <Dots points={TRAY_ART.screws} size={0.5} tone="mid" />
    </Lit>
    <Lit id="apart">
      <Line d={TRAY_ART.minor} tone="lo" />
      <Line d={TRAY_ART.major} tone="mid" />
      <Line d={TRAY_ART.datum} tone="mid" />
      <Dots points={TRAY_ART.gapMarks} size={0.6} tone="hi" />
    </Lit>
  </g>
);

const DECK = (
  <g key="deck">
    <Lit id="frame">
      <Solid paths={DECK_ART.slab} tone="mid" crease="faint" />
      <Line d={DECK_ART.inlay} tone="faint" />
      <Line d={DECK_ART.screwRings} tone="lo" />
      <Dots points={DECK_ART.screws} size={0.45} tone="mid" />
      <Solid paths={BOTTLE_ART.coaster} tone="mid" crease="faint" />
      <Line d={BOTTLE_ART.coasterRing} tone="faint" />
      <Dots points={BOTTLE_ART.coasterDots} size={0.45} tone="mid" />
      <Solid className="ml-bottle" paths={BOTTLE_ART.body} tone="hi" crease="faint" />
      <Line d={BOTTLE_ART.foot} tone="faint" />
      <Line d={BOTTLE_ART.level} tone="lo" />
      <Line d={BOTTLE_ART.shine} tone="mid" />
      <Solid className="ml-bottle" paths={BOTTLE_ART.shoulder} tone="hi" crease="faint" />
      <Solid paths={BOTTLE_ART.neck} tone="mid" crease="faint" />
      <Solid paths={BOTTLE_ART.collar} tone="mid" crease="faint" />
      <Solid paths={BOTTLE_ART.cap} tone="hi" crease="lo" />
      <Line d={BOTTLE_ART.capKnurl} tone="lo" />
      <Line d={BOTTLE_ART.capRing} tone="faint" />
      <Solid paths={BOTTLE_ART.bulb} tone="mid" crease="faint" />
      <Line d={BOTTLE_ART.bulbShine} tone="lo" />
      <Line d={BOTTLE_ART.bulbRing} tone="faint" />
      <Dots points={[BOTTLE_ART.tip]} size={0.45} tone="mid" />
    </Lit>
    <Lit id="fan">
      <Line d={DECK_ART.guide} tone="faint" dashed />
      {DECK_ART.stops.map((paths, index) => (
        <Solid key={index} paths={paths} tone="mid" crease="faint" />
      ))}
      <Dots points={DECK_ART.stopTops} size={0.45} tone="hi" />
      <Solid paths={DECK_ART.bandBack} tone="mid" crease="faint" />
      <FaceText transform={DECK_ART.label} size={4.2} tone="lo">
        merge
      </FaceText>
      <Solid paths={DECK_ART.washer} tone="lo" crease="faint" />
    </Lit>
  </g>
);

const FRONT = (
  <Lit key="front" id="fan">
    <Solid paths={DECK_ART.bandFront} tone="mid" crease="faint" />
    <Line d={DECK_ART.bandLine} tone="faint" />
    <Line d={DECK_ART.minor} tone="lo" />
    <Line d={DECK_ART.major} tone="mid" />
    <Dots points={DECK_ART.gapDots} size={0.55} tone="hi" />
    <Dots points={[...DECK_ART.ends, DECK_ART.seam]} size={0.45} tone="mid" />
    <Solid paths={DECK_ART.cap} tone="mid" crease="faint" />
    <Solid paths={DECK_ART.nut} tone="hi" crease="lo" />
    <Line d={DECK_ART.nutKnurl} tone="lo" />
    <Line d={DECK_ART.nutRing} tone="faint" />
    <Dots points={[DECK_ART.pin]} size={0.6} tone="hi" />
  </Lit>
);

export function LiquidArt() {
  return <LiquidLive label={LABEL} back={BACK} deck={DECK} front={FRONT} />;
}
