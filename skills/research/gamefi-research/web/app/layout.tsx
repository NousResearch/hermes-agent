import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "GameFi Research Workflow — Demo",
  description:
    "Showcase UI for the gamefi-research workflow: neutral, structured research on early-stage Web3 game projects. Research only — not financial advice.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
