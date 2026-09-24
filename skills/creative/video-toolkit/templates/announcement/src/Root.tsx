import React from "react";
import { Composition } from "remotion";
import { MainVideo } from "./MainVideo";

// --- CUSTOMIZE THESE ---
const VIDEO_CONFIG = {
  title: "Your Product Name",
  subtitle: "The tagline that sells it",
  features: [
    "Lightning fast performance",
    "Built for developers",
    "Open source",
  ],
  ctaText: "Try it today",
  ctaUrl: "github.com/your/repo",
  accentColor: "#00d4ff",
};

export const RemotionRoot: React.FC = () => (
  <>
    <Composition
      id="MainVideo"
      component={MainVideo}
      durationInFrames={900} // 30 seconds at 30fps
      fps={30}
      width={1920}
      height={1080}
      defaultProps={VIDEO_CONFIG}
    />
    {/* Social variant */}
    <Composition
      id="Social"
      component={MainVideo}
      durationInFrames={450} // 15 seconds
      fps={30}
      width={1080}
      height={1080}
      defaultProps={VIDEO_CONFIG}
    />
  </>
);
