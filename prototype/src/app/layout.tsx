import type { Metadata, Viewport } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--f-inter", display: "swap" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--f-jbmono", display: "swap" });
const grotesk = Space_Grotesk({ subsets: ["latin"], variable: "--f-grotesk", display: "swap" });

export const metadata: Metadata = {
  title: "Aro — Agent Workbench",
  description:
    "Aro Workbench prototype: one session, every surface — multi-agent threads, diff-centric review, parallel runs, skills and connectors. Aro Agent by samjuniors, based on Hermes Agent by Nous Research.",
  keywords: ["Aro", "Aro Agent", "Aro Desktop", "samjuniors", "AI agent", "Hermes Agent"],
  authors: [{ name: "samjuniors" }],
  icons: {
    icon: "/aro-icon.svg",
  },
  openGraph: {
    title: "Aro — Agent Workbench",
    description: "One session, every surface — the Aro Agent workbench by samjuniors",
    siteName: "Aro",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#06070A",
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" data-theme="obsidian" className={`${inter.variable} ${jetbrains.variable} ${grotesk.variable}`} suppressHydrationWarning>
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
