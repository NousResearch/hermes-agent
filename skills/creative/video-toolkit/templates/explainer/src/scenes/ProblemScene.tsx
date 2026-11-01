import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
} from "remotion";
import { loadFont } from "@remotion/google-fonts/Inter";

const { fontFamily } = loadFont();

interface ProblemSceneProps {
  problem: string;
  accentColor: string;
}

export const ProblemScene: React.FC<ProblemSceneProps> = ({
  problem,
  accentColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // "The Problem" label
  const labelOpacity = interpolate(frame, [0, fps * 0.3], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Problem text — word by word reveal
  const words = problem.split(" ");
  const framesPerWord = Math.floor(fps * 0.15);

  return (
    <>
      <AbsoluteFill style={{ background: "#0a0a0f" }} />

      {/* Subtle red gradient — problem = tension */}
      <AbsoluteFill
        style={{
          background: "radial-gradient(ellipse at 50% 80%, rgba(220,40,40,0.08), transparent 60%)",
        }}
      />

      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          fontFamily,
          padding: "0 200px",
        }}
      >
        {/* Label */}
        <div
          style={{
            fontSize: 18,
            fontWeight: 600,
            color: accentColor,
            opacity: labelOpacity,
            textTransform: "uppercase",
            letterSpacing: 4,
            marginBottom: 32,
          }}
        >
          The Problem
        </div>

        {/* Problem statement — word by word */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: 14,
            justifyContent: "center",
            maxWidth: 1200,
          }}
        >
          {words.map((word, i) => {
            const wordDelay = Math.floor(fps * 0.6) + i * framesPerWord;
            const wordEntrance = spring({
              frame: frame - wordDelay,
              fps,
              config: { damping: 14, mass: 0.5 },
            });

            return (
              <span
                key={i}
                style={{
                  fontSize: 56,
                  fontWeight: 600,
                  color: "white",
                  opacity: wordEntrance,
                  transform: `translateY(${interpolate(wordEntrance, [0, 1], [15, 0])}px)`,
                }}
              >
                {word}
              </span>
            );
          })}
        </div>
      </AbsoluteFill>
    </>
  );
};
