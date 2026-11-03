// zh-Hans Skills Hub (/zh-Hans/skills → /docs/zh-Hans/skills with the app-wide basePath).
import type { Metadata } from "next";
import { catalogAlternates } from "@/lib/catalog-routes";
import SkillsHub from "@/components/SkillsHub";

export const metadata: Metadata = {
  alternates: catalogAlternates("/skills", "zh-Hans"),
  title: "Skills Hub",
  description: "Browse all skills and plugins available for Hermes Agent",
};

export default function Page() {
  return <SkillsHub locale="zh-Hans" />;
}
