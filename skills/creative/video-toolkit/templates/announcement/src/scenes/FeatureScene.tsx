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

interface FeatureSceneProps {
  features: string[];
  accentColor: string;
}

export const FeatureScene: React.FC<FeatureSceneProps> = ({
  features,
  accentColor,
}) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  return (
    <>
      {/* Background — slightly different from title scene */}
      <AbsoluteFill
        style={{
          background: `radial-gradient(ellipse at 70% 40%, #0a1a3e, #0a0a1a 70%)`,
        }}
      />

      {/* Grid pattern overlay */}
      <AbsoluteFill
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)
          `,
          backgroundSize: "60px 60px",
        }}
      />

      {/* Features list */}
      <AbsoluteFill
        style={{
          justifyContent: "center",
          alignItems: "flex-start",
          padding: "0 180px",
          fontFamily,
        }}
      >
        {features.map((feature, i) => {
          const staggerDelay = Math.floor(fps * 0.8) * i;
          const featureFrame = frame - staggerDelay;

          // Entrance
          const entrance = spring({
            frame: featureFrame,
            fps,
            config: { damping: 14, mass: 0.5 },
          });

          // Accent dot
          const dotScale = spring({
            frame: featureFrame - 5,
            fps,
            config: { damping: 10, mass: 0.3, stiffness: 300 },
          });

          // Line underneath
          const lineWidth = interpolate(
            featureFrame,
            [0, fps * 0.5],
            [0, 100],
            {
              extrapolateLeft: "clamp",
              extrapolateRight: "clamp",
              easing: Easing.bezier(0.33, 1, 0.68, 1),
            }
          );

          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 24,
                marginBottom: 48,
                opacity: entrance,
                transform: `translateX(${interpolate(entrance, [0, 1], [40, 0])}px)`,
              }}
            >
              {/* Accent dot */}
              <div
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: "50%",
                  backgroundColor: accentColor,
                  transform: `scale(${dotScale})`,
                  boxShadow: `0 0 12px ${accentColor}`,
                  flexShrink: 0,
                }}
              />

              <div>
                <div
                  style={{
                    fontSize: 40,
                    fontWeight: 600,
                    color: "white",
                    letterSpacing: -0.5,
                  }}
                >
                  {feature}
                </div>
                {/* Accent underline */}
                <div
                  style={{
                    width: `${lineWidth}%`,
                    height: 2,
                    backgroundColor: accentColor,
                    opacity: 0.3,
                    marginTop: 8,
                    borderRadius: 1,
                  }}
                />
              </div>
            </div>
          );
        })}
      </AbsoluteFill>

      {/* Vignette */}
      <AbsoluteFill
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.5) 100%)",
          pointerEvents: "none",
        }}
      />
    </>
  );
};
