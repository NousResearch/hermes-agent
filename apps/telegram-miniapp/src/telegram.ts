export type TelegramThemeParams = {
  bg_color?: string;
  text_color?: string;
  hint_color?: string;
  link_color?: string;
  button_color?: string;
  button_text_color?: string;
  secondary_bg_color?: string;
  header_bg_color?: string;
  accent_text_color?: string;
  section_bg_color?: string;
  section_header_text_color?: string;
  subtitle_text_color?: string;
  destructive_text_color?: string;
};

export type TelegramWebAppUser = {
  id: number;
  first_name?: string;
  last_name?: string;
  username?: string;
  language_code?: string;
  is_premium?: boolean;
};

type TelegramWebApp = {
  initData?: string;
  initDataUnsafe?: {
    user?: TelegramWebAppUser;
    auth_date?: number;
    hash?: string;
  };
  themeParams?: TelegramThemeParams;
  colorScheme?: "light" | "dark";
  platform?: string;
  viewportHeight?: number;
  viewportStableHeight?: number;
  ready?: () => void;
  expand?: () => void;
};

declare global {
  interface Window {
    Telegram?: {
      WebApp?: TelegramWebApp;
    };
  }
}

export type TelegramRuntime = {
  isTelegram: boolean;
  initData: string;
  user?: TelegramWebAppUser;
  theme: TelegramThemeParams;
  colorScheme: "light" | "dark";
  platform: string;
};

export function prepareTelegramViewport(): void {
  const webApp = window.Telegram?.WebApp;
  webApp?.ready?.();
  webApp?.expand?.();
}

export function getTelegramRuntime(): TelegramRuntime {
  const webApp = window.Telegram?.WebApp;
  if (!webApp) {
    return {
      isTelegram: false,
      initData: "",
      theme: {},
      colorScheme: "dark",
      platform: "browser-mock",
    };
  }

  return {
    isTelegram: true,
    initData: webApp.initData ?? "",
    user: webApp.initDataUnsafe?.user,
    theme: webApp.themeParams ?? {},
    colorScheme: webApp.colorScheme ?? "dark",
    platform: webApp.platform ?? "telegram",
  };
}
