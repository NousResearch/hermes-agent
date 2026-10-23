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

interface TitleSceneProps {
  title: string;
  subtitle: string;
  accentColor: string;
}

export const TitleScene: React.FC<TitleSceneProps> = ({
  title,
  subtitle,
  accentColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // Title entrance — spring scale from center
  const titleScale = spring({
    frame,
    fps,
    config: { damping: 14, mass: 0.6, stiffness: 200 },
  });
  const titleOpacity = interpolate(frame, [0, fps * 0.3], [0, 1], {
    extrapolateRight: "clamp",
  });

  // Subtitle entrance — delayed fade + slide up
  const subtitleDelay = Math.floor(fps * 0.6);
  const subtitleOpacity = interpolate(
    frame - subtitleDelay,
    [0, fps * 0.4],
    [0, 1],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );
  const subtitleY = interpolate(
    frame - subtitleDelay,
    [0, fps * 0.4],
    [20, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  // Accent line — grows from center
  const lineWidth = interpolate(frame - Math.floor(fps * 0.4), [0, fps * 0.5], [0, 200], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  return (
    <>
      {/* Background */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 30% 50%, #1a0a2e, #0a0a1a 70%)`,
        }}
      />

      {/* Content */}
      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          fontFamily,
        }}
      >
        {/* Title */}
        <h1
          style={{
            fontSize: 80,
            fontWeight: 800,
            color: "white",
            margin: 0,
            transform: `scale(${titleScale})`,
            opacity: titleOpacity,
            letterSpacing: -2,
          }}
        >
          {title}
        </h1>

        {/* Accent line */}
        <div
          style={{
            width: lineWidth,
            height: 3,
            backgroundColor: accentColor,
            marginTop: 24,
            marginBottom: 24,
            borderRadius: 2,
            boxShadow: `0 0 12px ${accentColor}`,
          }}
        />

        {/* Subtitle */}
        <p
          style={{
            fontSize: 32,
            color: "rgba(255,255,255,0.6)",
            margin: 0,
            opacity: subtitleOpacity,
            transform: `translateY(${subtitleY}px)`,
            fontWeight: 400,
          }}
        >
          {subtitle}
        </p>
      </AbsoluteFill>

      {/* Vignette */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.6) 100%)",
          pointerEvents: "none",
        }}
      />
    </>
  );
};
