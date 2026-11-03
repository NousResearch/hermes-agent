// English Skills Hub (/skills → /docs/skills with the app-wide basePath).
import type { Metadata } from "next";
import { catalogAlternates } from "@/lib/catalog-routes";
import SkillsHub from "@/components/SkillsHub";

export const metadata: Metadata = {
  alternates: catalogAlternates("/skills", "en"),
  title: "Skills Hub",
  description: "Browse all skills and plugins available for Hermes Agent",
};

export default function Page() {
  return <SkillsHub locale="en" />;
}
