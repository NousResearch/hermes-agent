import React from "react";
import { TransitionSeries, linearTiming } from "@remotion/transitions";
import { fade } from "@remotion/transitions/fade";
import { TitleScene } from "./scenes/TitleScene";
import { FeatureScene } from "./scenes/FeatureScene";
import { CTAScene } from "./scenes/CTAScene";

interface VideoProps {
  title: string;
  subtitle: string;
  features: string[];
  ctaText: string;
  ctaUrl: string;
  accentColor: string;
}

export const MainVideo: React.FC<VideoProps> = (props) => (
  <TransitionSeries>
    <TransitionSeries.Sequence durationInFrames={270}>
      <TitleScene
        title={props.title}
        subtitle={props.subtitle}
        accentColor={props.accentColor}
      />
    </TransitionSeries.Sequence>

    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 20 })}
    />

    <TransitionSeries.Sequence durationInFrames={360}>
      <FeatureScene
        features={props.features}
        accentColor={props.accentColor}
      />
    </TransitionSeries.Sequence>

    <TransitionSeries.Transition
      presentation={fade()}
      timing={linearTiming({ durationInFrames: 20 })}
    />

    <TransitionSeries.Sequence durationInFrames={270}>
      <CTAScene
        ctaText={props.ctaText}
        ctaUrl={props.ctaUrl}
        accentColor={props.accentColor}
      />
    </TransitionSeries.Sequence>
  </TransitionSeries>
);
