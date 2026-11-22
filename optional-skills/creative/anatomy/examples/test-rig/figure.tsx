import { Plate } from "../../kit/react/draw";
import { StretchArt } from "./art";
import { readoutOf, PULLED } from "./rig";
import "./stretch.css";

export function TestRigFigure() {
  return (
    <Plate
      fig="Fig 6"
      title="Press and pull"
      hint="Press, then pull ← →"
      readout="120.0 × 32.0 · lean 0.0 px"
      keys={[
        { mark: "raised", label: "The glass while it is held" },
        { mark: "dotted", label: "Where it rests" },
        { mark: "edge", label: "The pull, from where I pressed to the finger" },
      ]}
      caption={`A 120 × 32 glass button on a test rig: a carriage on two guide rods held by four springs, a finger probe on a rail behind it, and a dial gauge that reads the lean. Pulled 80 px it leans ${readoutOf(PULLED).split("lean ")[1]}.`}
    >
      <StretchArt />
    </Plate>
  );
}
