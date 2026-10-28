// English Skills Hub (/skills → /docs/skills with the app-wide basePath).
import type { Metadata } from "next";
import SkillsHub from "@/components/SkillsHub";

export const metadata: Metadata = {
  title: "Skills Hub",
  description: "Browse all skills and plugins available for Hermes Agent",
};

export default function Page() {
  return <SkillsHub locale="en" />;
}
