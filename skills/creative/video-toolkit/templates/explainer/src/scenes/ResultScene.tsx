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

interface ResultSceneProps {
  result: string;
  accentColor: string;
}

export const ResultScene: React.FC<ResultSceneProps> = ({
  result,
  accentColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Result text entrance — dramatic scale from center
  const scale = spring({
    frame: frame - Math.floor(fps * 0.3),
    fps,
    config: { damping: 10, mass: 0.8, stiffness: 150 },
  });

  const opacity = interpolate(frame, [Math.floor(fps * 0.2), Math.floor(fps * 0.5)], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Glow pulse
  const glowIntensity = 8 + Math.sin(frame * 0.06) * 4;

  return (
    <>
      {/* Background — brightest scene, accent color presence */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 50% 50%, rgba(${parseInt(accentColor.slice(1, 3), 16)},${parseInt(accentColor.slice(3, 5), 16)},${parseInt(accentColor.slice(5, 7), 16)},0.1), #0a0a1a 60%)`,
        }}
      />

      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          fontFamily,
          padding: "0 200px",
          textAlign: "center",
        }}
      >
        {/* Label */}
        <div
          style={{
            fontSize: 18,
            fontWeight: 600,
            color: accentColor,
            textTransform: "uppercase",
            letterSpacing: 4,
            marginBottom: 32,
            opacity: interpolate(frame, [0, fps * 0.3], [0, 1], { extrapolateRight: "clamp" }),
          }}
        >
          The Result
        </div>

        {/* Result statement */}
        <h2
          style={{
            fontSize: 60,
            fontWeight: 800,
            color: "white",
            margin: 0,
            opacity,
            transform: `scale(${scale})`,
            letterSpacing: -1,
            textShadow: `0 0 ${glowIntensity}px rgba(255,255,255,0.15)`,
            maxWidth: 1000,
          }}
        >
          {result}
        </h2>
      </AbsoluteFill>

      {/* Vignette */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.7) 100%)",
          pointerEvents: "none",
        }}
      />
    </>
  );
};
