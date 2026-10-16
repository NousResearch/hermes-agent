import { Plate } from "../../kit/react/draw";
import { LiquidArt } from "./art";
import { GAPS, MERGE, RADII, bladesOutOf, joinedListOf, readoutOf } from "./rig";
import "./spot-plate.css";

const [LARGE, SMALL] = RADII;

export function SpotPlateFigure() {
  return (
    <Plate
      fig="Fig 7"
      title="Spot plate"
      hint="Slide across to open the fan"
      readout={readoutOf(MERGE)}
      keys={[
        { mark: "raised", label: `Joined: one piece of glass, R${LARGE} + R${SMALL}` },
        { mark: "edge", label: "The neck, where a pair runs together" },
        { mark: "lit", label: "The merge blade" },
      ]}
      caption={`A spot plate with six channels, each holding a pair of glass drops at gaps from ${GAPS[0]} to ${GAPS[GAPS.length - 1]} px. Each pair is drawn from one distance field, so a pair closer than merge runs together into one piece of glass. The fan sets merge: each feeler blade stops at its own gap. At ${MERGE} px the ${joinedListOf(MERGE)} pairs have joined, so ${bladesOutOf(MERGE)}.`}
    >
      <LiquidArt />
    </Plate>
  );
}
