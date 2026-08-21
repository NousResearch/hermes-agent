import React from "react";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { slide } from "@remotion/transitions/slide";
import { ProblemScene } from "./scenes/ProblemScene";
import { SolutionScene } from "./scenes/SolutionScene";
import { ResultScene } from "./scenes/ResultScene";

interface VideoProps {
  problem: string;
  solution: string;
  solutionPoints: string[];
  result: string;
  accentColor: string;
}

export const MainVideo: React.FC<VideoProps> = (props) => (
  <TransitionSeries>
    {/* Problem — establish the pain */}
    <TransitionSeries.Sequence durationInFrames={540}>
      <ProblemScene
        problem={props.problem}
        accentColor={props.accentColor}
      />
    </TransitionSeries.Sequence>

    <TransitionSeries.Transition
      presentation={slide({ direction: "from-right" })}
      timing={linearTiming({ durationInFrames: 25 })}
    />

    {/* Solution — the reveal */}
    <TransitionSeries.Sequence durationInFrames={720}>
      <SolutionScene
        solution={props.solution}
        points={props.solutionPoints}
        accentColor={props.accentColor}
      />
    </TransitionSeries.Sequence>

    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 20 })}
    />

    {/* Result — the payoff */}
    <TransitionSeries.Sequence durationInFrames={540}>
      <ResultScene
        result={props.result}
        accentColor={props.accentColor}
      />
    </TransitionSeries.Sequence>
  </TransitionSeries>
);
