import React from "react";
import { Composition } from "remotion";
import { MainVideo } from "./MainVideo";

// --- CUSTOMIZE THESE ---
const VIDEO_CONFIG = {
  problem: "Current solutions are slow, fragile, and hard to debug",
  solution: "A new approach that solves all three",
  solutionPoints: [
    "10x faster processing",
    "Self-healing architecture",
    "Built-in observability",
  ],
  result: "Teams ship 3x faster with 90% fewer incidents",
  accentColor: "#8b5cf6",
};

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="MainVideo"
      component={MainVideo}
      durationInFrames={1800} // 60 seconds at 30fps
      fps={30}
      width={1920}
      height={1080}
      defaultProps={VIDEO_CONFIG}
    />
  </>
);
