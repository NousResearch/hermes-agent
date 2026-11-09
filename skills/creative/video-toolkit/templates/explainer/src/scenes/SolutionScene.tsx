import React from "react";
import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  Easing,
} from "remotion";
import { loadFont } from "@remotion/google-fonts/Inter";

const { fontFamily } = loadFont();

interface SolutionSceneProps {
  solution: string;
  points: string[];
  accentColor: string;
}

export const SolutionScene: React.FC<SolutionSceneProps> = ({
  solution,
  points,
  accentColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Solution title entrance
  const titleEntrance = spring({
    frame,
    fps,
    config: { damping: 12, mass: 0.6 },
  });

  return (
    <>
      {/* Background — brighter, more hopeful */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 40% 40%, rgba(${parseInt(accentColor.slice(1, 3), 16)},${parseInt(accentColor.slice(3, 5), 16)},${parseInt(accentColor.slice(5, 7), 16)},0.06), #0a0a1a 60%)`,
        }}
      />

      <AbsoluteFill
        style={{
          justifyContent: "center",
          padding: "0 180px",
          fontFamily,
        }}
      >
        {/* Solution headline */}
        <h2
          style={{
            fontSize: 52,
            fontWeight: 700,
            color: "white",
            margin: 0,
            marginBottom: 48,
            opacity: titleEntrance,
            transform: `scale(${titleEntrance})`,
          }}
        >
          {solution}
        </h2>

        {/* Solution points */}
        {points.map((point, i) => {
          const pointDelay = Math.floor(fps * 1) + i * Math.floor(fps * 0.7);
          const entrance = spring({
            frame: frame - pointDelay,
            fps,
            config: { damping: 14, mass: 0.5 },
          });

          const lineProgress = interpolate(
            frame - pointDelay,
            [0, fps * 0.4],
            [0, 100],
            { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.bezier(0.33, 1, 0.68, 1) }
          );

          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 20,
                marginBottom: 36,
                opacity: entrance,
                transform: `translateX(${interpolate(entrance, [0, 1], [30, 0])}px)`,
              }}
            >
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  backgroundColor: accentColor,
                  boxShadow: `0 0 10px ${accentColor}`,
                  flexShrink: 0,
                }}
              />
              <span style={{ fontSize: 36, color: "rgba(255,255,255,0.9)", fontWeight: 500 }}>
                {point}
              </span>
            </div>
          );
        })}
      </AbsoluteFill>
    </>
  );
};
