import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { localizedHref, MARKETPLACE_URL, publicHref, type Locale } from './urls';

export function layoutOptions(locale: Locale): BaseLayoutProps {
  return {
    nav: {
      url: localizedHref('/', locale),
      title: <><img src={publicHref('/img/logo.png')} alt="" width={28} height={28} className="dark:hidden" /><img src={publicHref('/img/logo-dark.png')} alt="" width={28} height={28} className="hidden dark:block" /><span>Hermes Agent</span></>,
    },
    githubUrl: 'https://github.com/NousResearch/hermes-agent',
    links: [
      { text: 'Docs', url: localizedHref('/', locale) },
      { text: 'Skills', url: MARKETPLACE_URL.skills, external: true },
      { text: 'Plugins', url: MARKETPLACE_URL.plugins, external: true },
      { text: 'Download', url: 'https://hermes-agent.nousresearch.com/', external: true },
      { text: 'Discord', url: 'https://discord.gg/NousResearch', external: true },
    ],
    themeSwitch: { enabled: true },
  };
}
