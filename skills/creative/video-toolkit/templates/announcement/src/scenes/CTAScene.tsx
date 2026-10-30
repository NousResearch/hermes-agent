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

interface CTASceneProps {
  ctaText: string;
  ctaUrl: string;
  accentColor: string;
}

export const CTAScene: React.FC<CTASceneProps> = ({
  ctaText,
  ctaUrl,
  accentColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  // CTA text entrance
  const ctaEntrance = spring({
    frame,
    fps,
    config: { damping: 12, mass: 0.6 },
  });

  // URL entrance — delayed
  const urlDelay = Math.floor(fps * 0.5);
  const urlOpacity = interpolate(frame - urlDelay, [0, fps * 0.3], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const urlY = interpolate(frame - urlDelay, [0, fps * 0.3], [15, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // Glow pulse on the accent elements
  const glowIntensity = Math.sin(frame * 0.08) * 4 + 12;

  return (
    <>
      {/* Background — warmer for CTA */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 50% 60%, #1a0a2e, #0a0a1a 70%)`,
        }}
      />

      {/* Content */}
      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "center",
          fontFamily,
          gap: 32,
        }}
      >
        {/* CTA text */}
        <h2
          style={{
            fontSize: 64,
            fontWeight: 700,
            color: "white",
            margin: 0,
            transform: `scale(${ctaEntrance})`,
            letterSpacing: -1,
          }}
        >
          {ctaText}
        </h2>

        {/* URL */}
        <div
          style={{
            fontSize: 28,
            color: accentColor,
            opacity: urlOpacity,
            transform: `translateY(${urlY}px)`,
            fontWeight: 500,
            textShadow: `0 0 ${glowIntensity}px ${accentColor}`,
          }}
        >
          {ctaUrl}
        </div>
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
