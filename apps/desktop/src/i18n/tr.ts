import { defineLocale } from './define-locale'

const tr = defineLocale({
  common: {
    apply: 'Uygula',
    back: 'Geri',
    save: 'Kaydet',
    saving: 'Kaydediliyor',
    cancel: 'İptal',
    change: 'Değiştir',
    choose: 'Seç',
    clear: 'Temizle',
    close: 'Kapat',
    collapse: 'Daralt',
    confirm: 'Onayla',
    connect: 'Bağlan',
    connecting: 'Bağlanıyor',
    continue: 'Devam Et',
    copied: 'Kopyalandı',
    copy: 'Kopyala',
    copyFailed: 'Kopyalama başarısız',
    delete: 'Sil',
    docs: 'Belgeler',
    done: 'Bitti',
    error: 'Hata',
    failed: 'Başarısız',
    free: 'Ücretsiz',
    loading: 'Yükleniyor',
    notSet: 'Ayarlanmamış',
    refresh: 'Yenile',
    remove: 'Kaldır',
    replace: 'Değiştir',
    retry: 'Tekrar Dene',
    run: 'Çalıştır',
    send: 'Gönder',
    set: 'Ayarla',
    skip: 'Atla',
    update: 'Güncelle',
    on: 'Açık',
    off: 'Kapalı'
  },
  language: {
    label: 'Dil',
    description: 'Hermes arayüz dilini değiştirin',
    saving: 'Dil ayarı kaydediliyor',
    saveError: 'Dil ayarı kaydedilemedi',
    switchTo: 'Dile geç:',
    searchPlaceholder: 'Dil ara...',
    noResults: 'Dil bulunamadı'
  },
  fileMenu: {
    revealFinder: 'Bulucu\'da Göster',
    revealExplorer: 'Dosya Gezgini\'nde Göster',
    revealFileManager: 'Dosya Yöneticisi\'nde Göster',
    revealInSidebar: 'Kenar Çubuğunda Göster',
    copyPath: 'Yolu Kopyala',
    copyRelativePath: 'Göreli Yolu Kopyala',
    rename: 'Yeniden Adlandır',
    delete: 'Sil',
    renameTitle: 'Yeniden Adlandır',
    renameLabel: 'Yeni ad',
    pathCopied: 'Yol kopyalandı'
  },
  settings: {
    closeSettings: 'Ayarları Kapat',
    exportConfig: 'Yapılandırmayı Dışa Aktar',
    importConfig: 'Yapılandırmayı İçe Aktar',
    resetToDefaults: 'Varsayılana Sıfırla',
    nav: {
      providers: 'Sağlayıcılar',
      gateway: 'Ağ Geçidi',
      apiKeys: 'API Anahtarları',
      mcp: 'MCP',
      archivedChats: 'Arşivlenmiş Sohbetler',
      about: 'Hakkında',
      notifications: 'Bildirimler'
    }
  },
  composer: {
    message: 'Mesaj',
    stop: 'Durdur',
    send: 'Gönder',
    thinking: 'Düşünüyor',
    listening: 'Dinliyor',
    speaking: 'Konuşuyor',
    stopListening: 'Dinlemeyi Durdur',
    muteMic: 'Mikrofonu Kapat',
    unmuteMic: 'Mikrofonu Aç'
  },
  sidebar: {
    searchPlaceholder: 'Sohbet ara...',
    noMatch: 'Sonuç bulunamadı',
    pinned: 'Sabitlenmiş',
    sessions: 'Oturumlar'
  },
  notifications: {
    hide: 'Gizle',
    show: 'Göster',
    clearAll: 'Tümünü Temizle',
    dismiss: 'Kapat',
    details: 'Detaylar'
  },
  titlebar: {
    hideSidebar: 'Kenar Çubuğunu Gizle',
    showSidebar: 'Kenar Çubuğunu Göster',
    search: 'Ara',
    openSettings: 'Ayarları Aç',
    openKeybinds: 'Kısayolları Aç'
  },
  keybinds: {
    title: 'Klavye Kısayolları',
    reset: 'Sıfırla',
    resetAll: 'Tümünü Sıfırla',
    set: 'Ata'
  },
  boot: {
    ready: 'Hazır',
    steps: {
      connectingGateway: 'Ağ geçidine bağlanılıyor',
      loadingSettings: 'Ayarlar yükleniyor',
      loadingSessions: 'Oturumlar yükleniyor',
      startingHermesDesktop: 'Hermes Desktop başlatılıyor'
    },
    errors: {
      desktopBootFailed: 'Hermes Desktop başlatılamadı',
      gatewayConnectionLost: 'Ağ geçidi bağlantısı kesildi'
    }
  },
  updates: {
    updateNow: 'Şimdi Güncelle',
    maybeLater: 'Sonra',
    copy: 'Kopyala',
    copied: 'Kopyalandı',
    done: 'Tamam'
  },
  shell: {
    statusbar: {
      unknown: 'Bilinmiyor',
      gateway: 'Ağ Geçidi',
      gatewayReady: 'Hazır',
      gatewayOffline: 'Çevrimdışı'
    }
  },
  ui: {
    search: {
      clear: 'Temizle'
    }
  },
  agents: {
    close: 'Kapat',
    title: 'Ajanlar',
    running: 'Çalışıyor',
    failed: 'Başarısız',
    done: 'Tamamlandı',
    loading: 'Yükleniyor'
  },
  prompts: {
    gatewayDisconnected: 'Ağ geçidi bağlantısı kesildi'
  },
  desktop: {
    sessionUnavailable: 'Oturum kullanılamıyor',
    createSessionFailed: 'Oturum oluşturulamadı',
    promptFailed: 'İstek başarısız oldu',
    stopFailed: 'Durdurulamadı',
    archiveFailed: 'Arşivlenemedi',
    imageSaved: 'Görsel kaydedildi',
    downloadStarted: 'İndirme başladı'
  },
  cron: {
    close: 'Kapat',
    title: 'Zamanlanmış Görevler',
    search: 'Ara',
    loading: 'Yükleniyor'
  },
  artifacts: {
    search: 'Ara',
    refresh: 'Yenile',
    tabAll: 'Tümü',
    tabImages: 'Görseller',
    tabFiles: 'Dosyalar',
    tabLinks: 'Bağlantılar'
  },
  profiles: {
    close: 'Kapat',
    title: 'Profiller',
    search: 'Ara',
    loading: 'Yükleniyor',
    rename: 'Yeniden Adlandır',
    default: 'Varsayılan'
  },
  skills: {
    all: 'Tümü',
    loading: 'Yükleniyor',
    refresh: 'Yenile',
    searchSkills: 'Skill ara...'
  },
  errors: {
    reloadWindow: 'Pencereyi Yeniden Yükle',
    openLogs: 'Günlükleri Aç'
  },
  modelPicker: {
    title: 'Model Seçici',
    search: 'Ara',
    current: 'Geçerli'
  },
  starmap: {
    title: 'Yıldız Haritası',
    close: 'Kapat',
    refresh: 'Yenile',
    loading: 'Yükleniyor',
    memory: 'Bellek',
    copy: 'Kopyala',
    copied: 'Kopyalandı'
  },
  install: {
    error: 'Hata',
    copyCommand: 'Komutu Kopyala',
    viewDocs: 'Belgeleri Görüntüle'
  },
  messaging: {
    search: 'Ara',
    loading: 'Yükleniyor',
    enabled: 'Etkin',
    disabled: 'Devre Dışı',
    saving: 'Kaydediliyor',
    saved: 'Kaydedildi',
    openDocs: 'Belgeleri Aç'
  },
  assistant: {
    thread: {
      loadingSession: 'Oturum yükleniyor',
      loadingResponse: 'Yanıt yükleniyor',
      copy: 'Kopyala',
      refresh: 'Yenile',
      stop: 'Durdur',
      today: 'Bugün',
      yesterday: 'Dün',
      editMessage: 'Mesajı Düzenle',
      sendEdited: 'Düzenleneni Gönder'
    },
    approval: {
      run: 'Çalıştır',
      reject: 'Reddet',
      command: 'Komut',
      allowSession: 'Oturuma İzin Ver'
    },
    clarify: {
      other: 'Diğer',
      skip: 'Atla',
      placeholder: 'Cevabınız...'
    },
    tool: {
      code: 'Kod',
      copyCode: 'Kodu Kopyala',
      copyOutput: 'Çıktıyı Kopyala',
      copyCommand: 'Komutu Kopyala',
      outputAlt: 'Çıktı'
    }
  },
  onboarding: {
    startChatting: 'Sohbete Başla',
    change: 'Değiştir',
    free: 'Ücretsiz',
    copy: 'Kopyala',
    getKey: 'Anahtar Al',
    connected: 'Bağlandı',
    connecting: 'Bağlanıyor',
    recommended: 'Önerilen',
    chooseLater: 'Sonra Seç'
  }
})

export { tr }
