import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale } from './define-locale'

export const tr = defineLocale({
  common: {
    apply: 'Uygula',
    back: 'Geri',
    save: 'Kaydet',
    saving: 'Kaydediliyor…',
    cancel: 'İptal',
    change: 'Değiştir',
    choose: 'Seç',
    clear: 'Temizle',
    close: 'Kapat',
    collapse: 'Daralt',
    confirm: 'Onayla',
    connect: 'Bağlan',
    connecting: 'Bağlanıyor',
    continue: 'Devam et',
    copied: 'Kopyalandı',
    copy: 'Kopyala',
    copyFailed: 'Kopyalama başarısız',
    delete: 'Sil',
    docs: 'Belgeler',
    done: 'Tamam',
    error: 'Hata',
    failed: 'Başarısız',
    free: 'Ücretsiz',
    loading: 'Yükleniyor…',
    notSet: 'Ayarlanmadı',
    refresh: 'Yenile',
    remove: 'Kaldır',
    replace: 'Değiştir',
    retry: 'Yeniden dene',
    run: 'Çalıştır',
    send: 'Gönder',
    set: 'Ayarla',
    skip: 'Atla',
    update: 'Güncelle',
    on: 'Açık',
    off: 'Kapalı'
  },

  boot: {
    ready: 'Hermes Desktop hazır',
    desktopBootFailedWithMessage: message => `Masaüstü başlatılamadı: ${message}`,
    steps: {
      connectingGateway: 'Canlı masaüstü ağ geçidine bağlanıyor',
      loadingSettings: 'Hermes ayarları yükleniyor',
      loadingSessions: 'Son oturumlar yükleniyor',
      startingDesktopConnection: 'Masaüstü bağlantısı başlatılıyor',
      startingHermesDesktop: 'Hermes Desktop başlatılıyor…'
    },
    errors: {
      backgroundExited: 'Hermes arka plan işlemi sonlandı.',
      backgroundExitedDuringStartup: 'Hermes arka plan işlemi başlatma sırasında sonlandı.',
      backendStopped: 'Arka uç durdu',
      desktopBootFailed: 'Masaüstü başlatılamadı',
      gatewaySignInRequired: 'Ağ geçidi oturumu açma gerekli',
      ipcBridgeUnavailable: 'Masaüstü IPC köprüsü kullanılamıyor.'
    },
    failure: {
      title: 'Hermes başlatılamadı',
      description:
        'Arka plan ağ geçidi başlamadı. Aşağıdaki kurtarma adımlarından birini deneyin. Sohbetleriniz ve ayarlarınız silinmez.',
      remoteTitle: 'Uzak ağ geçidi oturumu açma gerekli',
      remoteDescription:
        'Uzak ağ geçidi oturumunuzun süresi doldu. Yeniden bağlanmak için tekrar oturum açın. Sohbetleriniz ve ayarlarınız silinmez.',
      retry: 'Yeniden dene',
      repairInstall: 'Kurulumu onar',
      useLocalGateway: 'Yerel ağ geçidini kullan',
      openLogs: 'Günlükleri aç',
      repairHint: 'Onarma yükleyiciyi yeniden çalıştırır; yeni bir makinede birkaç dakika sürebilir.',
      remoteSignInHint:
        'Ağ geçidi oturum açma penceresini açar. Bunun yerine paketlenmiş arka uca geçmek için "Yerel ağ geçidini kullan" seçeneğini kullanın.',
      hideRecentLogs: 'Son günlükleri gizle',
      showRecentLogs: 'Son günlükleri göster',
      signedInTitle: 'Oturum açıldı',
      signedInMessage: 'Uzak ağ geçidine yeniden bağlanılıyor…',
      signInIncompleteTitle: 'Oturum açma tamamlanmadı',
      signInIncompleteMessage: 'Oturum açma penceresi kimlik doğrulama tamamlanmadan kapandı.',
      signInFailed: 'Oturum açma başarısız',
      signInToRemoteGateway: 'Uzak ağ geçidinde oturum aç',
      signInWithProvider: provider => `${provider} ile oturum aç`,
      identityProvider: 'kimlik sağlayıcınız'
    }
  },

  notifications: {
    region: 'Bildirimler',
    hide: 'Gizle',
    show: 'Göster',
    more: count => `${count} bildirim daha`,
    clearAll: 'Tümünü temizle',
    dismiss: 'Bildirimi kapat',
    details: 'Ayrıntılar',
    copyDetail: 'Ayrıntıyı kopyala',
    copyDetailFailed: 'Bildirim ayrıntısı kopyalanamadı',
    backendOutOfDateTitle: 'Arka uç güncel değil',
    backendOutOfDateMessage:
      'Hermes arka ucunuz bu masaüstü sürümünden daha eski ve düzgün çalışmayabilir. Hizalamak için güncelleyin.',
    updateHermes: "Hermes'i güncelle",
    updateReadyTitle: 'Güncelleme hazır',
    updateReadyMessage: count => `${count} yeni değişiklik mevcut.`,
    seeWhatsNew: 'Yeniliklere bak',
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs STT için ELEVENLABS_API_KEY gereklidir.',
      elevenLabsRejectedKey: 'ElevenLabs API anahtarını reddetti (401).',
      methodNotAllowed:
        'Masaüstü arka ucu bu isteği reddetti (405 Method Not Allowed). Hermes Desktop\'ı yeniden başlatmayı deneyin.',
      microphonePermission: 'Mikrofon izni reddedildi.',
      openaiRejectedApiKey: 'OpenAI API anahtarını reddetti.',
      openaiRejectedApiKeyWithStatus: status => `OpenAI API anahtarını reddetti (${status} invalid_api_key).`,
      openaiTtsNeedsKey: 'OpenAI TTS için VOICE_TOOLS_OPENAI_KEY veya OPENAI_API_KEY gereklidir.'
    },
    voice: {
      configureSpeechToText: 'Ses modunu kullanmak için konuşmadan metne dönüştürmeyi yapılandırın.',
      couldNotStartSession: 'Ses oturumu başlatılamadı',
      microphoneAccessDenied: 'Mikrofon erişimi reddedildi.',
      microphoneConstraintsUnsupported: 'Bu cihaz mikrofon kısıtlamalarını desteklemiyor.',
      microphoneFailed: 'Mikrofon başarısız oldu',
      microphoneInUse: 'Mikrofon başka bir uygulama tarafından kullanılıyor.',
      microphonePermissionDenied: 'Mikrofon izni reddedildi.',
      microphoneStartFailed: 'Mikrofon kaydı başlatılamadı.',
      microphoneUnsupported: 'Bu çalışma zamanı mikrofon kaydını desteklemiyor.',
      noMicrophone: 'Mikrofon bulunamadı.',
      noSpeechDetected: 'Konuşma algılanamadı',
      playbackFailed: 'Ses çalma başarısız',
      recordingFailed: 'Ses kaydı başarısız',
      transcriptionFailed: 'Ses dökümü başarısız',
      transcriptionUnavailable: 'Ses dökümü henüz kullanılamıyor.',
      tryRecordingAgain: 'Tekrar kaydetmeyi deneyin.',
      unavailable: 'Ses kullanılamıyor'
    },
    native: {
      approvalTitle: 'Onay gerekli',
      approveAction: 'Onayla',
      rejectAction: 'Reddet',
      inputTitle: 'Giriş gerekli',
      inputBody: 'Hermes yanıtınızı bekliyor.',
      turnDoneTitle: 'Hermes tamamladı',
      turnDoneBody: 'Yanıt hazır.',
      turnErrorTitle: 'Tur başarısız',
      backgroundDoneTitle: 'Arka plan görevi tamamlandı',
      backgroundFailedTitle: 'Arka plan görevi başarısız'
    }
  },

  titlebar: {
    hideSidebar: 'Kenar çubuğunu gizle',
    showSidebar: 'Kenar çubuğunu göster',
    search: 'Ara',
    searchTitle: 'Oturumları, görünümleri ve eylemleri ara',
    swapSidebarSides: 'Kenar çubuğu taraflarını değiştir',
    swapSidebarSidesTitle: 'Oturumlar ve dosya tarayıcısı taraflarını değiştir',
    hideRightSidebar: 'Sağ kenar çubuğunu gizle',
    showRightSidebar: 'Sağ kenar çubuğunu göster',
    muteHaptics: 'Dokunsal geri bildirimi kapat',
    unmuteHaptics: 'Dokunsal geri bildirimi aç',
    openSettings: 'Ayarları aç',
    openKeybinds: 'Klavye kısayolları'
  },

  keybinds: {
    title: 'Klavye kısayolları',
    subtitle: open => `Yeniden atamak için bir kısayola tıklayın · ${open} bu paneli yeniden açar.`,
    rebind: 'Yeniden ata',
    reset: 'Varsayılana sıfırla',
    resetAll: 'Tümünü sıfırla',
    pressKey: 'Bir tuşa basın…',
    set: 'ayarlandı',
    conflictWith: label => `"${label}" ile çakışıyor`,
    categories: {
      composer: 'Düzenleyici',
      profiles: 'Profiller',
      session: 'Oturum',
      navigation: 'Gezinme',
      view: 'Görünüm'
    },
    actions: {
      'keybinds.openPanel': 'Klavye kısayollarını aç',
      'nav.commandPalette': 'Komut paletini aç',
      'nav.commandCenter': 'Komut merkezini aç',
      'nav.settings': 'Ayarları aç',
      'nav.profiles': 'Profilleri aç',
      'nav.skills': 'Becerileri aç',
      'nav.messaging': 'Mesajlaşmayı aç',
      'nav.artifacts': 'Yapıtları aç',
      'nav.cron': 'Zamanlanmış görevleri aç',
      'nav.agents': 'Ajanları aç',
      'session.new': 'Yeni oturum',
      'session.next': 'Sonraki oturum',
      'session.prev': 'Önceki oturum',
      'session.slot.1': "Son oturum 1'e geç",
      'session.slot.2': "Son oturum 2'ye geç",
      'session.slot.3': "Son oturum 3'e geç",
      'session.slot.4': "Son oturum 4'e geç",
      'session.slot.5': "Son oturum 5'e geç",
      'session.slot.6': "Son oturum 6'ya geç",
      'session.slot.7': "Son oturum 7'ye geç",
      'session.slot.8': "Son oturum 8'e geç",
      'session.slot.9': "Son oturum 9'a geç",
      'session.focusSearch': 'Oturumları ara',
      'session.togglePin': 'Mevcut oturumu sabitle / sabitlemeni kaldır',
      'composer.focus': 'Düzenleyiciye odaklan',
      'composer.modelPicker': 'Model seçiciyi aç',
      'view.toggleSidebar': 'Oturum kenar çubuğunu aç/kapat',
      'view.toggleRightSidebar': 'Dosya tarayıcısını aç/kapat',
      'view.showFiles': 'Dosya tarayıcısını göster',
      'view.showTerminal': 'Terminali göster',
      'view.terminalSelection': 'Terminal seçimini düzenleyiciye gönder',
      'view.closePreviewTab': 'Önizleme sekmesini kapat',
      'view.flipPanes': 'Kenar çubuğu taraflarını değiştir',
      'appearance.toggleMode': 'Açık / koyu modunu değiştir',
      'profile.default': 'Varsayılan profile geç',
      'profile.switch.1': "Profil 1'e geç",
      'profile.switch.2': "Profil 2'ye geç",
      'profile.switch.3': "Profil 3'e geç",
      'profile.switch.4': "Profil 4'e geç",
      'profile.switch.5': "Profil 5'e geç",
      'profile.switch.6': "Profil 6'ya geç",
      'profile.switch.7': "Profil 7'ye geç",
      'profile.switch.8': "Profil 8'e geç",
      'profile.switch.9': "Profil 9'a geç",
      'profile.switch.10': "Profil 10'a geç",
      'profile.switch.11': "Profil 11'e geç",
      'profile.switch.12': "Profil 12'ye geç",
      'profile.switch.13': "Profil 13'e geç",
      'profile.switch.14': "Profil 14'e geç",
      'profile.switch.15': "Profil 15'e geç",
      'profile.switch.16': "Profil 16'ya geç",
      'profile.switch.17': "Profil 17'ye geç",
      'profile.switch.18': "Profil 18'e geç",
      'profile.next': 'Sonraki profil',
      'profile.prev': 'Önceki profil',
      'profile.toggleAll': 'Tüm profiller görünümünü aç/kapat',
      'profile.create': 'Profil oluştur',
      'composer.send': 'Mesaj gönder',
      'composer.newline': 'Yeni satır ekle',
      'composer.steer': 'Çalışan turu yönlendir',
      'composer.sendQueued': 'Sıradaki turu gönder',
      'composer.mention': "Dosyaları, klasörleri, URL'leri referansla",
      'composer.slash': 'Eğik çizgi komut paleti',
      'composer.help': 'Hızlı yardım',
      'composer.history': 'Açılır pencere / geçmişi döngüle',
      'composer.cancel': 'Açılır pencereyi kapat · çalışmayı iptal et'
    }
  },

  language: {
    label: 'Dil',
    description: 'Masaüstü arayüzü için dil seçin.',
    saving: 'Dil kaydediliyor…',
    saveError: 'Dil güncellemesi başarısız',
    switchTo: 'Dil değiştir',
    searchPlaceholder: 'Dil ara…',
    noResults: 'Dil bulunamadı'
  },

  settings: {
    closeSettings: 'Ayarları kapat',
    exportConfig: 'Yapılandırmayı dışa aktar',
    importConfig: 'Yapılandırmayı içe aktar',
    resetToDefaults: 'Varsayılanlara sıfırla',
    resetConfirm: 'Tüm ayarlar Hermes varsayılanlarına sıfırlansın mı?',
    exportFailed: 'Dışa aktarma başarısız',
    resetFailed: 'Sıfırlama başarısız',
    nav: {
      providers: 'Sağlayıcılar',
      providerAccounts: 'Hesaplar',
      providerApiKeys: 'API Anahtarları',
      gateway: 'Ağ Geçidi',
      apiKeys: 'Araçlar ve Anahtarlar',
      keysTools: 'Araçlar',
      keysSettings: 'Ayarlar',
      mcp: 'MCP',
      archivedChats: 'Arşivlenmiş Sohbetler',
      about: 'Hakkında',
      notifications: 'Bildirimler'
    },
    notifications: {
      title: 'Bildirimler',
      intro:
        'Uygulama içi bildirimlerden ayrı, yerel masaüstü bildirimleri. Bunlar cihaza özgüdür; her bilgisayar kendi ayarlarını korur.',
      enableAll: 'Bildirimleri etkinleştir',
      enableAllDesc: 'Ana anahtar. Bunu kapatmak aşağıdaki tüm bildirimleri devre dışı bırakır.',
      focusedHint: "Tamamlanma uyarıları yalnızca Hermes arka plandayken çalışır.",
      kinds: {
        approval: {
          label: 'Onay gerekli',
          description: 'Bir komut onaylamanızı ya da reddetmenizi bekliyor.'
        },
        input: {
          label: 'Giriş gerekli',
          description: 'Hermes bir soru sordu veya parola ya da gizli bilgi istiyor.'
        },
        turnDone: {
          label: 'Yanıt hazır',
          description: "Hermes arka plandayken bir tur tamamlandı."
        },
        turnError: {
          label: 'Tur başarısız',
          description: 'Bir tur hatayla sona erdi.'
        },
        backgroundDone: {
          label: 'Arka plan görevi tamamlandı',
          description: 'Arka planda çalışan bir terminal komutu tamamlandı.'
        }
      },
      test: 'Test bildirimi gönder',
      testTitle: 'Hermes',
      testBody: 'Bildirimler çalışıyor.',
      testSent:
        "Test gönderildi. Hiçbir şey görünmüyorsa işletim sistemi bildirim izinlerini ve Odaklanma/Rahatsız Etme modunu kontrol edin.",
      testUnsupported: 'Bu sistem yerel bildirimleri desteklemiyor.',
      completionSoundTitle: 'Tamamlanma Sesi',
      completionSoundDesc: 'Ajan turu bittiğinde çalar. Bir ön ayar seçin ve burada dinleyin.',
      completionSoundPreview: 'Dinle'
    },
    sections: {
      model: 'Model',
      chat: 'Sohbet',
      appearance: 'Görünüm',
      workspace: 'Çalışma Alanı',
      safety: 'Güvenlik',
      memory: 'Bellek ve Bağlam',
      voice: 'Ses',
      advanced: 'Gelişmiş'
    },
    searchPlaceholder: {
      about: 'Hermes Desktop Hakkında',
      config: 'Ayarlarda ara...',
      gateway: 'Ağ geçidi bağlantısı...',
      keys: 'API anahtarlarında ara...',
      mcp: 'MCP sunucularında ara...',
      sessions: 'Arşivlenmiş oturumlarda ara...'
    },
    modeOptions: {
      light: { label: 'Açık', description: 'Parlak masaüstü yüzeyleri' },
      dark: { label: 'Koyu', description: 'Göz yormayan çalışma alanı' },
      system: { label: 'Sistem', description: 'İşletim sistemi görünümünü takip et' }
    },
    appearance: {
      title: 'Görünüm',
      intro:
        'Bunlar yalnızca masaüstüne özgü görüntü tercihleridir. Mod parlaklığı; tema ise vurgu paletini ve sohbet yüzeyi stilini kontrol eder.',
      colorMode: 'Renk Modu',
      colorModeDesc: "Sabit bir mod seçin veya Hermes'in sistem ayarınızı takip etmesine izin verin.",
      toolViewTitle: 'Araç Çağrısı Görünümü',
      toolViewDesc: 'Ürün görünümü ham araç verilerini gizler; Teknik görünüm tam girdi/çıktıyı gösterir.',
      translucencyTitle: 'Pencere Yarı Saydamlığı',
      translucencyDesc: 'Pencerenin tamamı aracılığıyla masaüstünüzü görün. Yalnızca macOS ve Windows.',
      product: 'Ürün',
      productDesc: 'Kısa özetlerle kullanıcı dostu araç etkinliği.',
      technical: 'Teknik',
      technicalDesc: 'Ham araç argümanları/sonuçları ve düşük düzey ayrıntıları içerir.',
      themeTitle: 'Tema',
      themeDesc: 'Yalnızca masaüstü paletleri. Seçilen mod üstte uygulanır.',
      themeProfileNote: profile => `"${profile}" profili için kaydedildi — her profil kendi temasını korur.`,
      installTitle: "VS Code'dan Yükle",
      installDesc:
        'Renk temasını masaüstü paletine dönüştürmek için bir Marketplace uzantı kimliği yapıştırın (örn. dracula-theme.theme-dracula).',
      installPlaceholder: 'yayıncı.uzantı',
      installButton: 'Yükle',
      installing: 'Yükleniyor…',
      installError: 'Bu tema yüklenemedi.',
      installed: name => `"${name}" yüklendi.`,
      removeTheme: 'Temayı kaldır',
      importedBadge: 'İçe aktarıldı'
    },
    fieldLabels: defineFieldCopy({
      model: 'Varsayılan Model',
      modelContextLength: 'Bağlam Penceresi',
      fallbackProviders: 'Yedek Model',
      toolsets: 'Etkin Araç Setleri',
      timezone: 'Saat Dilimi',
      display: {
        personality: 'Kişilik',
        showReasoning: 'Muhakeme Bloğu'
      },
      agent: {
        maxTurns: 'Maksimum Ajan Adımı',
        imageInputMode: 'Görsel Eki',
        apiMaxRetries: 'API Yeniden Deneme',
        serviceTier: 'Hizmet Katmanı',
        toolUseEnforcement: 'Araç Kullanım Zorunluluğu'
      },
      terminal: {
        cwd: 'Çalışma Dizini',
        backend: 'Çalıştırma Arka Ucu',
        timeout: 'Komut Zaman Aşımı',
        persistentShell: 'Kalıcı Kabuk',
        envPassthrough: 'Ortam Değişkeni Geçişi',
        dockerImage: 'Docker Görüntüsü',
        singularityImage: 'Singularity Görüntüsü',
        modalImage: 'Modal Görüntüsü',
        daytonaImage: 'Daytona Görüntüsü'
      },
      fileReadMaxChars: 'Dosya Okuma Sınırı',
      toolOutput: {
        maxBytes: 'Terminal Çıktı Sınırı',
        maxLines: 'Dosya Sayfa Sınırı',
        maxLineLength: 'Satır Uzunluğu Sınırı'
      },
      codeExecution: {
        mode: 'Kod Yürütme Modu'
      },
      approvals: {
        mode: 'Onay Modu',
        timeout: 'Onay Zaman Aşımı',
        mcpReloadConfirm: 'MCP Yeniden Yükleme Onayı'
      },
      commandAllowlist: 'Komut İzin Listesi',
      security: {
        redactSecrets: 'Gizli Bilgileri Gizle',
        allowPrivateUrls: "Özel URL'lere İzin Ver"
      },
      browser: {
        allowPrivateUrls: "Tarayıcı Özel URL'leri",
        autoLocalForPrivateUrls: "Özel URL'ler için Yerel Tarayıcıyı Kullan"
      },
      checkpoints: {
        enabled: 'Dosya Kontrol Noktaları',
        maxSnapshots: 'Kontrol Noktası Sınırı'
      },
      voice: {
        recordKey: 'Ses Kısayolu',
        maxRecordingSeconds: 'Maksimum Kayıt Süresi',
        autoTts: 'Yanıtları Sesle Oku'
      },
      stt: {
        enabled: 'Konuşmadan Metne',
        provider: 'STT Sağlayıcısı',
        local: {
          model: 'Yerel Döküm Modeli',
          language: 'Döküm Dili'
        },
        openai: {
          model: 'OpenAI STT Modeli'
        },
        groq: {
          model: 'Groq STT Modeli'
        },
        mistral: {
          model: 'Mistral STT Modeli'
        },
        elevenlabs: {
          modelId: 'ElevenLabs STT Modeli',
          languageCode: 'ElevenLabs Dili',
          tagAudioEvents: 'Ses Olaylarını Etiketle',
          diarize: 'Konuşmacı Ayrıştırma'
        }
      },
      tts: {
        provider: 'Metinden Sese Sağlayıcısı',
        edge: {
          voice: 'Edge Sesi'
        },
        openai: {
          model: 'OpenAI TTS Modeli',
          voice: 'OpenAI Sesi'
        },
        elevenlabs: {
          voiceId: 'ElevenLabs Sesi',
          modelId: 'ElevenLabs Modeli'
        },
        xai: {
          voiceId: 'xAI (Grok) Sesi',
          language: 'xAI Dili'
        },
        minimax: {
          model: 'MiniMax TTS Modeli',
          voiceId: 'MiniMax Sesi'
        },
        mistral: {
          model: 'Mistral TTS Modeli',
          voiceId: 'Mistral Sesi'
        },
        gemini: {
          model: 'Gemini TTS Modeli',
          voice: 'Gemini Sesi'
        },
        neutts: {
          model: 'NeuTTS Modeli',
          device: 'NeuTTS Cihazı'
        },
        kittentts: {
          model: 'KittenTTS Modeli',
          voice: 'KittenTTS Sesi'
        },
        piper: {
          voice: 'Piper Sesi'
        }
      },
      memory: {
        memoryEnabled: 'Kalıcı Bellek',
        userProfileEnabled: 'Kullanıcı Profili',
        memoryCharLimit: 'Bellek Bütçesi',
        userCharLimit: 'Profil Bütçesi',
        provider: 'Bellek Sağlayıcısı'
      },
      context: {
        engine: 'Bağlam Motoru'
      },
      compression: {
        enabled: 'Otomatik Sıkıştırma',
        threshold: 'Sıkıştırma Eşiği',
        targetRatio: 'Sıkıştırma Hedefi',
        protectLastN: 'Korunan Son Mesajlar'
      },
      delegation: {
        model: 'Alt Ajan Modeli',
        provider: 'Alt Ajan Sağlayıcısı',
        maxIterations: 'Alt Ajan Tur Sınırı',
        maxConcurrentChildren: 'Paralel Alt Ajanlar',
        childTimeoutSeconds: 'Alt Ajan Zaman Aşımı',
        reasoningEffort: 'Alt Ajan Muhakeme Yoğunluğu'
      },
      updates: {
        nonInteractiveLocalChanges: 'Uygulama İçi Güncelleme Sırasında Yerel Değişiklikler'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Düzenleyicide başka bir model seçilmedikçe yeni sohbetlerde kullanılır.',
      modelContextLength: '0 bırakılırsa seçili modelden algılanan bağlam penceresi kullanılır.',
      fallbackProviders:
        'Varsayılan model başarısız olduğunda denecek sağlayıcı:model biçimindeki yedekler.',
      display: {
        personality: 'Yeni oturumlar için varsayılan asistan stili.',
        showReasoning: 'Arka uç muhakeme içeriği sağladığında gösterir.'
      },
      timezone:
        "Hermes yerel zaman bağlamına ihtiyaç duyduğunda kullanılır. Boş bırakılırsa sistem saat dilimi kullanılır.",
      agent: {
        imageInputMode: 'Görsel eklerinin modele nasıl gönderileceğini kontrol eder.',
        maxTurns: "Hermes'in tek bir çalıştırmada durmadan önce yapabileceği maksimum araç çağrısı turu."
      },
      terminal: {
        cwd: 'Araçlar ve terminal çalışmaları için varsayılan proje klasörü.',
        persistentShell: 'Arka uç destekliyorsa komutlar arasında kabuk durumunu korur.',
        envPassthrough: 'Araç yürütmeye aktarılacak ortam değişkenleri.'
      },
      codeExecution: {
        mode: 'Kod yürütmenin mevcut projeyle ne kadar kısıtlı olacağını ayarlar.'
      },
      fileReadMaxChars: "Hermes'in tek bir dosya okumasında alabileceği maksimum karakter sayısı.",
      approvals: {
        mode: "Hermes'in açık onay gerektiren komutları nasıl ele alacağını ayarlar.",
        timeout: 'Onay isteminin zaman aşımına uğramadan önce beklenecek süre.'
      },
      security: {
        redactSecrets: 'Algılanan gizli bilgileri mümkün olan her yerde modelden gizler.'
      },
      checkpoints: {
        enabled: 'Dosya düzenlemelerinden önce geri alma için anlık görüntüler oluşturur.'
      },
      memory: {
        memoryEnabled: 'Gelecekteki oturumlar için yararlı kalıcı bellekler kaydeder.',
        userProfileEnabled: 'Kullanıcı tercihlerini özetleyen kısa bir profil tutar.'
      },
      context: {
        engine: 'Uzun sohbetler bağlam sınırına yaklaştığında yönetim stratejisi.'
      },
      compression: {
        enabled: 'Sohbet büyüdüğünde eski bağlamı özetler.'
      },
      voice: {
        autoTts: 'Asistan yanıtlarını otomatik olarak sesle okur.'
      },
      stt: {
        enabled: 'Yerel veya sağlayıcı tabanlı konuşma döküm özelliğini etkinleştirir.',
        elevenlabs: {
          languageCode:
            'İsteğe bağlı ISO-639-3 dil kodu. Boş bırakılırsa ElevenLabs otomatik algılar.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          "Uygulamadan Hermes güncellenirken yerel kaynak değişikliklerinin korunup korunmayacağını seçin. Terminal güncellemelerinde her zaman sorulur."
      }
    }),
    about: {
      heading: 'Hermes Desktop',
      version: value => `Sürüm ${value}`,
      versionUnavailable: 'Sürüm bilgisi alınamadı',
      updates: 'Güncellemeler',
      checkNow: 'Şimdi kontrol et',
      checking: 'Kontrol ediliyor…',
      seeWhatsNew: 'Yeniliklere bak',
      releaseNotes: 'Sürüm notları',
      onLatest: 'En son sürümdesiniz.',
      installing: 'Bir güncelleme yükleniyor.',
      cantUpdate: 'Bu yapı uygulamanın içinden kendini güncelleyemiyor.',
      cantReach: 'Güncelleme sunucusuna ulaşılamadı.',
      tapCheck: '"Şimdi kontrol et"e tıklayarak güncelleme arayın.',
      updateReady: count => `Yeni bir güncelleme hazır (${count} değişiklik dahil).`,
      lastChecked: age => `Son kontrol: ${age}`,
      justNowSuffix: ' · az önce',
      automaticUpdates: 'Otomatik güncellemeler',
      automaticUpdatesDesc:
        "Hermes arka planda otomatik olarak güncellemeleri kontrol eder ve hazır olduğunda sizi bilgilendirir.",
      branchCommit: (branch, commit) => `Dal ${branch} · Commit ${commit}`,
      never: 'hiç',
      justNow: 'az önce',
      minAgo: count => `${count} dakika önce`,
      hoursAgo: count => `${count} saat önce`,
      daysAgo: count => `${count} gün önce`
    },
    config: {
      none: 'Yok',
      noneParen: '(yok)',
      notSet: 'Ayarlanmadı',
      commaSeparated: 'virgülle ayrılmış değerler',
      loading: 'Hermes yapılandırması yükleniyor...',
      emptyTitle: 'Yapılandırılacak bir şey yok',
      emptyDesc: 'Bu bölümde ayarlanabilir seçenek bulunmuyor.',
      failedLoad: 'Ayarlar yüklenemedi',
      autosaveFailed: 'Otomatik kaydetme başarısız',
      imported: 'Yapılandırma içe aktarıldı',
      invalidJson: "Geçersiz yapılandırma JSON'ı"
    },
    credentials: {
      pasteKey: 'Anahtarı yapıştır',
      pasteLabelKey: label => `${label} anahtarını yapıştır`,
      optional: 'İsteğe bağlı',
      enterValueFirst: 'Önce bir değer girin.',
      couldNotSave: 'Kimlik bilgisi kaydedilemedi.',
      remove: 'Kaldır',
      or: 'veya',
      escToCancel: 'iptal için Esc',
      getKey: 'Anahtar al',
      saving: 'Kaydediliyor'
    },
    envActions: {
      actionsFor: label => `${label} için eylemler`,
      credentialActions: 'Kimlik bilgisi eylemleri',
      docs: 'Belgeler',
      hideValue: 'Değeri gizle',
      revealValue: 'Değeri göster',
      replace: 'Değiştir',
      set: 'Ayarla',
      clear: 'Temizle'
    },
    gateway: {
      loading: 'Ağ geçidi ayarları yükleniyor...',
      unavailableTitle: 'Ağ geçidi ayarları kullanılamıyor',
      unavailableDesc: 'Masaüstü IPC köprüsü ağ geçidi ayarlarını sunmuyor.',
      title: 'Ağ Geçidi Bağlantısı',
      envOverride: 'ortam değişkeni geçişi',
      intro:
        "Hermes Desktop varsayılan olarak kendi yerel ağ geçidini başlatır. Bu uygulamanın başka bir makinede veya güvenilir bir proxy arkasında çalışmakta olan Hermes arka ucunu kontrol etmesini istiyorsanız uzak ağ geçidini kullanın. Kendi uzak ana bilgisayarını ayarlamak için aşağıdan bir profil seçin.",
      appliesTo: 'Uygulandığı yer',
      allProfiles: 'Tüm profiller',
      defaultConnection: 'Kendi geçişi olmayan her profil için varsayılan bağlantı.',
      profileConnection: profile =>
        `Yalnızca "${profile}" etkin profil olduğunda kullanılan bağlantı. Varsayılanı devralması için Yerel olarak ayarlayın.`,
      envOverrideTitle: 'Ortam değişkenleri bu masaüstü oturumunu kontrol ediyor.',
      envOverrideDesc:
        "Aşağıdaki kayıtlı ayarı kullanmak için HERMES_DESKTOP_REMOTE_URL ve HERMES_DESKTOP_REMOTE_TOKEN'ı kaldırın.",
      localTitle: 'Yerel ağ geçidi',
      localDesc: "Localhost'ta özel bir Hermes arka ucu başlatır. Bu varsayılandır ve çevrimdışı çalışır.",
      remoteTitle: 'Uzak ağ geçidi',
      remoteDesc:
        'Bu masaüstü kabuğunu uzak bir Hermes arka ucuna bağlar. Barındırılan ağ geçitleri OAuth veya kullanıcı adı ve parola kullanır; kendi barındırdığınız olanlar oturum jetonu kullanabilir.',
      remoteUrlTitle: 'Uzak URL',
      remoteUrlDesc: 'Uzak pano arka ucunun temel URL\'si. /hermes gibi yol önekleri desteklenir.',
      probing: 'Bu ağ geçidinin kimlik doğrulama yöntemi kontrol ediliyor…',
      probeError:
        "Bu ağ geçidine henüz ulaşılamıyor. URL'yi kontrol edin — yanıt verdiğinde kimlik doğrulama yöntemi görünecektir.",
      signedIn: 'Oturum açıldı',
      signIn: 'Oturum aç',
      signOut: 'Oturumu kapat',
      signInWith: provider => `${provider} ile oturum aç`,
      authTitle: 'Kimlik Doğrulama',
      authSignedInPassword:
        'Bu ağ geçidi kullanıcı adı ve parola kullanıyor. Oturum açtınız; oturum otomatik olarak yenilenir.',
      authSignedInOauth:
        'Bu ağ geçidi OAuth kullanıyor. Oturum açtınız; oturum otomatik olarak yenilenir.',
      authNeedsPassword:
        'Bu ağ geçidi kullanıcı adı ve parola kullanıyor. Bu masaüstü uygulamasını yetkilendirmek için oturum açın.',
      authNeedsOauth: provider =>
        `Bu ağ geçidi OAuth kullanıyor. Bu masaüstü uygulamasını yetkilendirmek için ${provider} ile oturum açın.`,
      tokenTitle: 'Oturum jetonu',
      tokenDesc:
        'REST ve WebSocket erişimi için kullanılan pano oturum jetonu. Kayıtlı jetonu korumak için boş bırakın.',
      existingToken: value => `Mevcut jeton ${value}`,
      savedToken: 'kaydedildi',
      pasteSessionToken: 'Oturum jetonunu yapıştır',
      testRemote: 'Uzağı test et',
      saveForRestart: 'Sonraki yeniden başlatma için kaydet',
      saveAndReconnect: 'Kaydet ve yeniden bağlan',
      diagnostics: 'Tanılama',
      diagnosticsDesc:
        "Dosya yöneticinizde desktop.log'u göster — ağ geçidi başlatılamadığında yararlıdır.",
      openLogs: 'Günlükleri aç',
      incompleteTitle: 'Uzak ağ geçidi eksik',
      incompleteSignIn: 'Uzaka geçmeden önce uzak bir URL girin ve oturum açın.',
      incompleteToken: 'Uzaka geçmeden önce uzak URL ve oturum jetonu girin.',
      incompleteSignInTest: 'Test etmeden önce uzak URL girin ve oturum açın.',
      incompleteTokenTest: 'Test etmeden önce uzak URL ve oturum jetonu girin.',
      enterUrlFirst: 'Önce uzak bir URL girin.',
      restartingTitle: 'Ağ geçidi bağlantısı yeniden başlatılıyor',
      savedTitle: 'Ağ geçidi ayarları kaydedildi',
      restartingMessage: 'Hermes Desktop, kayıtlı ayarları kullanarak yeniden bağlanacak.',
      savedMessage: 'Sonraki yeniden başlatma için kaydedildi.',
      connectedTo: (baseUrl, version) => `${baseUrl}${version ? ` · Hermes ${version}` : ''} bağlandı`,
      reachableTitle: 'Uzak ağ geçidine ulaşılabildi',
      signedOutTitle: 'Oturum kapatıldı',
      signedOutMessage: 'Uzak ağ geçidi oturumu temizlendi.',
      failedLoad: 'Ağ geçidi ayarları yüklenemedi',
      signInFailed: 'Oturum açma başarısız',
      signOutFailed: 'Oturum kapatma başarısız',
      testFailed: 'Uzak ağ geçidi testi başarısız',
      applyFailed: 'Ağ geçidi ayarları uygulanamadı',
      saveFailed: 'Ağ geçidi ayarları kaydedilemedi'
    },
    keys: {
      loading: 'API anahtarları ve kimlik bilgileri yükleniyor...',
      failedLoad: 'API anahtarları yüklenemedi',
      empty: 'Bu kategoride henüz yapılandırılmış bir şey yok.'
    },
    mcp: {
      loading: 'MCP sunucuları yükleniyor...',
      failedLoad: 'MCP yapılandırması yüklenemedi',
      nameRequiredTitle: 'Ad gerekli',
      nameRequiredMessage: 'Bu MCP sunucusuna bir yapılandırma anahtarı verin.',
      objectRequired: 'Sunucu yapılandırması bir JSON nesnesi olmalıdır',
      invalidJson: "Geçersiz MCP JSON'ı",
      saveFailed: 'Kaydetme başarısız',
      removeFailed: 'Kaldırma başarısız',
      gatewayUnavailableTitle: 'Ağ geçidi kullanılamıyor',
      gatewayUnavailableMessage: "MCP'yi yeniden yüklemeden önce ağ geçidini yeniden bağlayın.",
      reloadedTitle: 'MCP araçları yeniden yüklendi',
      reloadedMessage: 'Yeni araç şemaları yeni turlara uygulanır.',
      reloadFailed: 'MCP yeniden yükleme başarısız',
      savedTitle: 'MCP sunucusu kaydedildi',
      savedMessage: name => `${name}, MCP yeniden yüklemesinin ardından uygulanır.`,
      newServer: 'Yeni sunucu',
      reload: "MCP'yi yeniden yükle",
      reloading: 'Yeniden yükleniyor...',
      emptyTitle: 'MCP sunucusu yok',
      emptyDesc: 'MCP araçlarını sunmak için bir stdio veya HTTP sunucusu ekleyin.',
      disabled: 'devre dışı',
      editServer: 'Sunucuyu düzenle',
      name: 'Ad',
      serverJson: "Sunucu JSON'ı",
      remove: 'Kaldır',
      saveServer: 'Sunucuyu kaydet'
    },
    model: {
      loading: 'Model yapılandırması yükleniyor...',
      appliesDesc:
        'Yeni oturumlara uygulanır. Etkin sohbeti anında değiştirmek için düzenleyicideki model seçiciyi kullanın.',
      provider: 'Sağlayıcı',
      model: 'Model',
      applying: 'Uygulanıyor...',
      auxiliaryTitle: 'Yardımcı modeller',
      resetAllToMain: 'Tümünü ana modele sıfırla',
      auxiliaryDesc:
        'Yardımcı görevler varsayılan olarak ana modelde çalışır. Geçersiz kılmak için herhangi bir göreve özel bir model atayın.',
      setToMain: 'Ana modele ayarla',
      change: 'Değiştir',
      autoUseMain: 'otomatik · ana modeli kullan',
      providerDefault: '(sağlayıcı varsayılanı)',
      tasks: {
        vision: { label: 'Görüntü', hint: 'Görsel analizi' },
        web_extract: { label: 'Web çıkarma', hint: 'Sayfa özetleme' },
        compression: { label: 'Sıkıştırma', hint: 'Bağlam sıkıştırma' },
        skills_hub: { label: 'Beceri merkezi', hint: 'Beceri arama' },
        approval: { label: 'Onay', hint: 'Akıllı otomatik onay' },
        mcp: { label: 'MCP', hint: 'MCP araç yönlendirme' },
        title_generation: { label: 'Başlık oluşturma', hint: 'Oturum başlıkları' },
        curator: { label: 'Küratör', hint: 'Beceri kullanım incelemesi' }
      }
    },
    providers: {
      connectAccount: 'Hesap bağla',
      haveApiKey: 'Bunun yerine API anahtarınız mı var?',
      intro:
        'Abonelikle oturum açın — API anahtarı kopyalamanız gerekmez. Hermes, tarayıcı oturumunu sizin için burada çalıştırır.',
      connected: 'Bağlandı',
      collapse: 'Daralt',
      connectAnother: 'Başka bir sağlayıcı bağla',
      otherProviders: 'Diğer sağlayıcılar',
      removeConfirm: provider => `${provider} kaldırılsın mı?`,
      removeExternal: (provider, command) =>
        `${provider} Hermes dışında yönetiliyor. ${command} ile kaldırın.`,
      removeKeyManaged: provider =>
        `${provider} bir API anahtarıyla yapılandırıldı. API Anahtarları'ndan kaldırın.`,
      removedTitle: 'Hesap kaldırıldı',
      removedMessage: provider => `${provider} kaldırıldı.`,
      failedRemove: provider => `${provider} kaldırılamadı`,
      noProviderKeys: 'Kullanılabilir sağlayıcı API anahtarı yok.',
      loading: 'Sağlayıcılar yükleniyor...'
    },
    sessions: {
      loading: 'Arşivlenmiş oturumlar yükleniyor…',
      archivedTitle: 'Arşivlenmiş oturumlar',
      archivedIntro:
        'Arşivlenmiş sohbetler kenar çubuğunda gizlenir ancak tüm mesajlarını korur. Arşivlemek için kenar çubuğundaki bir sohbete Ctrl/⌘ tıklayın.',
      emptyArchivedTitle: 'Arşivlenmiş öğe yok',
      emptyArchivedDesc: 'Burada gizlemek için bir sohbeti arşivleyin.',
      unarchive: 'Arşivden çıkar',
      deletePermanently: 'Kalıcı olarak sil',
      messages: count => `${count} mesaj`,
      restored: 'Geri yüklendi',
      deleteConfirm: title => `"${title}" kalıcı olarak silinsin mi? Bu işlem geri alınamaz.`,
      defaultDirTitle: 'Varsayılan proje dizini',
      defaultDirDesc:
        'Başka bir klasör seçilmedikçe yeni oturumlar bu klasörde başlar. Ayarlanmazsa giriş dizininiz kullanılır.',
      defaultDirUpdated:
        'Varsayılan proje dizini güncellendi — geçerli olması için yeni bir sohbet başlatın (Ctrl/⌘+N)',
      defaultsTo: label => `Varsayılan: ${label}.`,
      change: 'Değiştir',
      choose: 'Seç',
      clear: 'Temizle',
      notSet: 'Ayarlanmadı',
      failedLoad: 'Arşivlenmiş oturumlar yüklenemedi',
      unarchiveFailed: 'Arşivden çıkarma başarısız',
      deleteFailed: 'Silme başarısız',
      updateDirFailed: 'Varsayılan dizin güncellenemedi',
      clearDirFailed: 'Varsayılan dizin temizlenemedi'
    },
    toolsets: {
      loadingConfig: 'Yapılandırma yükleniyor',
      savedTitle: 'Kimlik bilgisi kaydedildi',
      savedMessage: key => `${key} güncellendi.`,
      removedTitle: 'Kimlik bilgisi kaldırıldı',
      removedMessage: key => `${key} kaldırıldı.`,
      failedSave: key => `${key} kaydedilemedi`,
      failedRemove: key => `${key} kaldırılamadı`,
      failedReveal: key => `${key} gösterilemedi`,
      removeConfirm: key => `${key}, .env dosyasından kaldırılsın mı?`,
      set: 'Ayarlandı',
      notSet: 'Ayarlanmadı',
      selectedTitle: 'Sağlayıcı seçildi',
      selectedMessage: provider => `${provider} artık etkin.`,
      failedSelect: provider => `${provider} seçilemedi`,
      failedLoad: 'Araç yapılandırması yüklenemedi',
      noProviderOptions:
        'Bu araç setinin sağlayıcı seçeneği yok — etkinleştirin ve mevcut kurulumunuzla çalışır.',
      noProviders: 'Bu araç seti için şu an kullanılabilir sağlayıcı yok.',
      ready: 'Hazır',
      nousIncluded: "Nous aboneliğine dahildir — etkinleştirmek için Nous Portal'da oturum açın.",
      noApiKeyRequired: 'API anahtarı gerekmez.',
      postSetupHint: step =>
        `Bu arka uç tek seferlik bir kurulum gerektirir (${step}). Bu makinede çalışır — birkaç dakika sürebilir.`,
      postSetupRun: 'Kurulumu çalıştır',
      postSetupRunning: 'Yükleniyor…',
      postSetupStarting: 'Başlatılıyor…',
      postSetupCompleteTitle: 'Kurulum tamamlandı',
      postSetupCompleteMessage: step => `${step} yüklendi.`,
      postSetupErrorTitle: 'Kurulum hatalarla tamamlandı',
      postSetupErrorMessage: step => `${step} günlüğünü kontrol edin.`,
      postSetupFailed: step => `${step} kurulumu çalıştırılamadı`
    }
  },

  skills: {
    tabSkills: 'Beceriler',
    tabToolsets: 'Araç Setleri',
    all: 'Tümü',
    searchSkills: 'Becerilerde ara...',
    searchToolsets: 'Araç setlerinde ara...',
    refresh: 'Becerileri yenile',
    refreshing: 'Beceriler yenileniyor',
    loading: 'Özellikler yükleniyor...',
    noSkillsTitle: 'Beceri bulunamadı',
    noSkillsDesc: 'Daha geniş bir arama veya farklı bir kategori deneyin.',
    noToolsetsTitle: 'Araç seti bulunamadı',
    noToolsetsDesc: 'Daha geniş bir arama sorgusu deneyin.',
    noDescription: 'Açıklama yok.',
    configured: 'Yapılandırıldı',
    needsKeys: 'Anahtar gerekiyor',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} araç seti etkin`,
    configureToolset: label => `${label} yapılandır`,
    toggleToolset: label => `${label} araç setini aç/kapat`,
    skillsLoadFailed: 'Beceriler yüklenemedi',
    toolsetsRefreshFailed: 'Araç setleri yenilenemedi',
    skillEnabled: 'Beceri etkinleştirildi',
    skillDisabled: 'Beceri devre dışı bırakıldı',
    toolsetEnabled: 'Araç seti etkinleştirildi',
    toolsetDisabled: 'Araç seti devre dışı bırakıldı',
    appliesToNewSessions: name => `${name} yeni oturumlara uygulanır.`,
    failedToUpdate: name => `${name} güncellenemedi`
  },

  agents: {
    close: 'Ajanları kapat',
    title: 'Oluşturma ağacı',
    subtitle: 'Mevcut tur için canlı alt ajan etkinliği.',
    emptyTitle: 'Canlı alt ajan yok',
    emptyDesc: 'Bir tur iş devredildiğinde, alt ajanlar ilerlemelerini buraya yayınlar.',
    running: 'Çalışıyor',
    failed: 'Başarısız',
    done: 'Tamamlandı',
    streaming: 'Yayınlanıyor',
    files: 'Dosyalar',
    moreFiles: count => `+${count} dosya daha`,
    delegation: index => `Devir ${index}`,
    workers: count => `${count} çalışan`,
    workersActive: count => `${count} etkin`,
    agentsCount: count => `${count} ajan`,
    activeCount: count => `${count} etkin`,
    failedCount: count => `${count} başarısız`,
    toolsCount: count => `${count} araç`,
    filesCount: count => `${count} dosya`,
    updatedAgo: age => `${age} önce güncellendi`,
    ageNow: 'şimdi',
    ageSeconds: seconds => `${seconds}sn önce`,
    ageMinutes: minutes => `${minutes}dk önce`,
    ageHours: hours => `${hours}sa önce`,
    durationSeconds: seconds => `${seconds}sn`,
    durationMinutes: (minutes, seconds) => `${minutes}dk ${seconds}sn`,
    tokensK: k => `${k}k tok`,
    tokens: value => `${value} tok`
  },

  commandCenter: {
    close: 'Komut merkezini kapat',
    paletteTitle: 'Komut paleti',
    back: 'Geri',
    searchPlaceholder: 'Oturumları, görünümleri ve eylemleri ara',
    goTo: 'Git',
    goToSession: 'Oturuma git',
    commandCenter: 'Komut Merkezi',
    appearance: 'Görünüm',
    settings: 'Ayarlar',
    changeTheme: 'Tema değiştir...',
    changeColorMode: 'Renk modunu değiştir...',
    installTheme: {
      title: 'Tema yükle...',
      placeholder: 'VS Code Marketplace\'te ara...',
      loading: 'Marketplace aranıyor...',
      error: "Marketplace'e ulaşılamadı.",
      empty: 'Eşleşen tema yok.',
      install: 'Yükle',
      installing: 'Yükleniyor...',
      installed: 'Yüklendi',
      installs: count => `${count} yükleme`
    },
    settingsFields: 'Ayar alanları',
    mcpServers: 'MCP sunucuları',
    archivedChats: 'Arşivlenmiş sohbetler',
    sections: { sessions: 'Oturumlar', system: 'Sistem', usage: 'Kullanım' },
    sectionDescriptions: {
      sessions: 'Oturumları ara ve yönet',
      system: 'Durum, günlükler ve sistem eylemleri',
      usage: 'Zaman içinde jeton, maliyet ve beceri etkinliği'
    },
    nav: {
      newChat: { title: 'Yeni oturum', detail: 'Yeni bir oturum başlat' },
      settings: { title: 'Ayarlar', detail: 'Hermes masaüstünü yapılandır' },
      skills: {
        title: 'Beceriler ve Araçlar',
        detail: 'Becerileri, araç setlerini ve sağlayıcıları etkinleştir'
      },
      messaging: {
        title: 'Mesajlaşma',
        detail: 'Telegram, Slack, Discord ve daha fazlasını ayarla'
      },
      artifacts: { title: 'Yapıtlar', detail: 'Oluşturulan çıktılara göz at' }
    },
    sectionEntries: {
      sessions: { title: 'Oturumlar paneli', detail: 'Oturumları ara, sabitle ve yönet' },
      system: {
        title: 'Sistem paneli',
        detail: 'Ağ geçidi durumu, günlükler, yeniden başlatma/güncelleme'
      },
      usage: { title: 'Kullanım paneli', detail: 'Jeton, maliyet ve beceri etkinliği' }
    },
    providerNavigate: 'Gezin',
    providerSessions: 'Oturumlar',
    refresh: 'Yenile',
    refreshing: 'Yenileniyor...',
    noResults: 'Eşleşen sonuç bulunamadı.',
    pinSession: 'Oturumu sabitle',
    unpinSession: 'Oturumu sabitlemeden çıkar',
    exportSession: 'Oturumu dışa aktar',
    deleteSession: 'Oturumu sil',
    noSessions: 'Henüz oturum yok.',
    gatewayRunning: 'Mesajlaşma ağ geçidi çalışıyor',
    gatewayStopped: 'Mesajlaşma ağ geçidi durdu',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Etkin oturum ${count}`,
    restartMessaging: 'Mesajlaşmayı yeniden başlat',
    updateHermes: "Hermes'i güncelle",
    actionRunning: 'çalışıyor',
    actionDone: 'tamamlandı',
    actionFailed: 'başarısız',
    actionStartedWaiting: 'Eylem başlatıldı, durum bekleniyor...',
    loadingStatus: 'Durum yükleniyor...',
    recentLogs: 'Son günlükler',
    noLogs: 'Henüz günlük yüklenmedi.',
    days: count => `${count}g`,
    statSessions: 'Oturumlar',
    statApiCalls: 'API çağrıları',
    statTokens: 'Jeton giriş/çıkış',
    statCost: 'Tahmini maliyet',
    actualCost: cost => `gerçek ${cost}`,
    loadingUsage: 'Kullanım yükleniyor...',
    noUsage: period => `Son ${period} günde kullanım yok.`,
    retry: 'Yeniden dene',
    dailyTokens: 'Günlük jetonlar',
    input: 'giriş',
    output: 'çıkış',
    noDailyActivity: 'Günlük etkinlik yok.',
    topModels: 'En çok kullanılan modeller',
    noModelUsage: 'Henüz model kullanımı yok.',
    topSkills: 'En çok kullanılan beceriler',
    noSkillActivity: 'Henüz beceri etkinliği yok.',
    actions: count => `${count} eylem`
  },

  messaging: {
    search: 'Mesajlaşmada ara...',
    loading: 'Mesajlaşma platformları yükleniyor...',
    loadFailed: 'Mesajlaşma platformları yüklenemedi',
    states: {
      connected: 'Bağlandı',
      connecting: 'Bağlanıyor',
      disabled: 'Devre dışı',
      fatal: 'Hata',
      gateway_stopped: 'Mesajlaşma ağ geçidi durdu',
      not_configured: 'Kurulum gerekli',
      pending_restart: 'Yeniden başlatma gerekli',
      retrying: 'Yeniden deneniyor',
      startup_failed: 'Başlatma başarısız'
    },
    unknown: 'Bilinmiyor',
    hintPendingRestart: 'Bu değişikliği uygulamak için durum çubuğundan ağ geçidini yeniden başlatın.',
    hintGatewayStopped: 'Bağlanmak için durum çubuğundan ağ geçidini başlatın.',
    credentialsSet: 'Kimlik bilgileri ayarlandı',
    needsSetup: 'Kurulum gerekli',
    gatewayStopped: 'Mesajlaşma ağ geçidi durdu',
    getCredentials: 'Kimlik bilgilerinizi alın',
    openSetupGuide: 'Kurulum kılavuzunu aç',
    required: 'Gerekli',
    recommended: 'Önerilen',
    advanced: count => `Gelişmiş (${count})`,
    noTokenNeeded:
      'Bu platform burada jeton gerektirmiyor. Yukarıdaki kurulum kılavuzunu kullanın, ardından aşağıdan etkinleştirin.',
    enabled: 'Etkin',
    disabled: 'Devre dışı',
    unsavedChanges: 'Kaydedilmemiş değişiklikler',
    saving: 'Kaydediliyor...',
    saveChanges: 'Değişiklikleri kaydet',
    saved: 'Kaydedildi',
    replaceValue: 'Mevcut değeri değiştir',
    openDocs: 'Belgeleri aç',
    clearField: key => `${key} temizle`,
    enableAria: name => `${name} etkinleştir`,
    disableAria: name => `${name} devre dışı bırak`,
    platformEnabled: name => `${name} etkinleştirildi`,
    platformDisabled: name => `${name} devre dışı bırakıldı`,
    restartToApply: 'Bu değişikliğin geçerli olması için ağ geçidini yeniden başlatın.',
    setupSaved: name => `${name} kurulumu kaydedildi`,
    restartToReconnect: 'Yeni kimlik bilgileriyle yeniden bağlanmak için ağ geçidini yeniden başlatın.',
    keyCleared: key => `${key} temizlendi`,
    setupUpdated: name => `${name} kurulumu güncellendi.`,
    failedUpdate: name => `${name} güncellenemedi`,
    failedSave: name => `${name} kaydedilemedi`,
    failedClear: key => `${key} temizlenemedi`,
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Bot jetonu',
        help: '@BotFather ile bir bot oluşturun, ardından verdiği jetonu yapıştırın.',
        placeholder: 'Telegram bot jetonunu yapıştır'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'İzin verilen Telegram kullanıcı kimlikleri',
        help: "Önerilen. @userinfobot'tan virgülle ayrılmış sayısal kimlikler. Bu olmadan herkes botunuza DM atabilir."
      },
      TELEGRAM_PROXY: {
        label: 'Proxy URL\'si',
        help: "Yalnızca Telegram'ın engellendiği ağlarda gereklidir."
      },
      DISCORD_BOT_TOKEN: {
        label: 'Bot jetonu',
        help: "Discord Geliştirici Portalı'nda bir uygulama oluşturun, bot ekleyin, ardından jetonunu yapıştırın."
      },
      DISCORD_ALLOWED_USERS: {
        label: 'İzin verilen Discord kullanıcı kimlikleri',
        help: 'Önerilen. Virgülle ayrılmış Discord kullanıcı kimlikleri.'
      },
      DISCORD_REPLY_TO_MODE: { label: 'Yanıt stili', help: 'first, all veya off.' },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Tüm Discord kullanıcılarına izin ver',
        help: 'Yalnızca geliştirme amaçlı. true olduğunda izin listesi olmadan herkes bota DM atabilir.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'Ana kanal kimliği',
        help: 'Botun proaktif mesajlar gönderdiği kanal (cron çıktısı, hatırlatıcılar).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Ana kanal adı',
        help: 'Günlükler ve durum çıktısındaki ana kanalın görünen adı.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Tüm iMessage kullanıcılarına izin ver',
        help: 'true olduğunda BlueBubbles izin listesini atlar.'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: 'Tüm Mattermost kullanıcılarına izin ver' },
      MATTERMOST_HOME_CHANNEL: { label: 'Ana kanal' },
      QQ_ALLOW_ALL_USERS: { label: 'Tüm QQ kullanıcılarına izin ver' },
      QQBOT_HOME_CHANNEL: {
        label: 'QQ ana kanalı',
        help: 'Cron teslimatı için varsayılan kanal veya grup.'
      },
      QQBOT_HOME_CHANNEL_NAME: { label: 'QQ ana kanal adı' },
      SLACK_BOT_TOKEN: {
        label: 'Slack bot jetonu',
        help: "Slack uygulamanızı yükledikten sonra OAuth & Permissions'daki bot jetonunu kullanın.",
        placeholder: 'Slack bot jetonunu yapıştır'
      },
      SLACK_APP_TOKEN: {
        label: 'Slack uygulama jetonu',
        help: 'Socket Mode için gereken uygulama düzeyinde jetonu kullanın.',
        placeholder: 'Slack uygulama jetonunu yapıştır'
      },
      SLACK_ALLOWED_USERS: {
        label: 'İzin verilen Slack kullanıcı kimlikleri',
        help: 'Önerilen. Virgülle ayrılmış Slack kullanıcı kimlikleri.'
      },
      MATTERMOST_URL: { label: 'Sunucu URL\'si', placeholder: 'https://mattermost.example.com' },
      MATTERMOST_TOKEN: { label: 'Bot jetonu' },
      MATTERMOST_ALLOWED_USERS: {
        label: 'İzin verilen kullanıcı kimlikleri',
        help: 'Önerilen. Virgülle ayrılmış Mattermost kullanıcı kimlikleri.'
      },
      MATRIX_HOMESERVER: { label: 'Sunucu URL\'si', placeholder: 'https://matrix.org' },
      MATRIX_ACCESS_TOKEN: { label: 'Erişim jetonu' },
      MATRIX_USER_ID: { label: 'Bot kullanıcı kimliği', placeholder: '@hermes:example.org' },
      MATRIX_ALLOWED_USERS: {
        label: 'İzin verilen Matrix kullanıcı kimlikleri',
        help: 'Önerilen. @kullanıcı:sunucu biçiminde virgülle ayrılmış kullanıcı kimlikleri.'
      },
      SIGNAL_HTTP_URL: {
        label: 'Signal köprü URL\'si',
        placeholder: 'http://127.0.0.1:8080',
        help: 'Çalışan bir signal-cli REST köprüsünün URL\'si.'
      },
      SIGNAL_ACCOUNT: {
        label: 'Telefon numarası',
        help: 'signal-cli köprüsüne kayıtlı numara.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'İzin verilen Signal kullanıcıları',
        help: 'Önerilen. Virgülle ayrılmış Signal tanımlayıcıları.'
      },
      WHATSAPP_ENABLED: {
        label: 'WhatsApp köprüsünü etkinleştir',
        help: 'Aşağıdaki geçiş tarafından otomatik olarak ayarlanır. Bilmiyorsanız değiştirmeyin.'
      },
      WHATSAPP_MODE: { label: 'Köprü modu' },
      WHATSAPP_ALLOWED_USERS: {
        label: 'İzin verilen WhatsApp kullanıcıları',
        help: 'Önerilen. Virgülle ayrılmış telefon numaraları veya WhatsApp kimlikleri.'
      }
    },
    platformIntro: {}
  },

  profiles: {
    close: 'Profilleri kapat',
    nameHint: 'Küçük harfler, rakamlar, tireler ve alt çizgiler. Bir harf veya rakamla başlamalıdır.',
    title: 'Profiller',
    count: count => `${count} profil`,
    loading: 'Profiller yükleniyor...',
    newProfile: 'Yeni profil',
    allProfiles: 'Tüm profiller',
    showAllProfiles: 'Tüm profilleri göster',
    switchToProfile: name => `${name} profiline geç`,
    manageProfiles: 'Profilleri yönet...',
    actionsFor: name => `${name} için eylemler`,
    color: 'Renk...',
    colorFor: name => `${name} rengi`,
    setColor: color => `${color} rengini ayarla`,
    autoColor: 'Otomatik',
    noProfiles: 'Henüz profil yok.',
    selectPrompt: 'Ayrıntılarını görmek için bir profil seçin.',
    refresh: 'Profilleri yenile',
    refreshing: 'Profiller yenileniyor',
    default: 'varsayılan',
    skills: count => `${count} beceri`,
    env: 'ortam',
    defaultBadge: 'Varsayılan',
    rename: 'Yeniden adlandır',
    copySetup: 'Kurulumu kopyala',
    copying: 'Kopyalanıyor...',
    modelLabel: 'Model',
    skillsLabel: 'Beceriler',
    notSet: 'Ayarlanmadı',
    soulDesc: 'Bu profile yerleşik sistem istemi ve kişilik talimatları.',
    soulOptional: 'isteğe bağlı',
    soulPlaceholder: mode =>
      `Bu profil için sistem istemi / kişilik.\nBoş bırakırsanız ${mode} varsayılanı kullanılır.`,
    soulPlaceholderCloned: 'klonlandı',
    soulPlaceholderEmpty: 'boş',
    unsavedChanges: 'Kaydedilmemiş değişiklikler',
    loadingSoul: 'SOUL.md yükleniyor...',
    emptySoul: 'Boş SOUL.md — kişilik yazmaya başlayın...',
    saving: 'Kaydediliyor...',
    saveSoul: "SOUL.md'yi kaydet",
    deleteTitle: 'Profil silinsin mi?',
    deleteDescPrefix: 'Bu işlem ',
    deleteDescMid: ' öğesini siler ve onun ',
    deleteDescSuffix: ' dizinini kaldırır. Bu işlem geri alınamaz.',
    deleting: 'Siliniyor...',
    createDesc: 'Profiller bağımsız Hermes ortamlarıdır: ayrı yapılandırma, beceriler ve SOUL.md.',
    nameLabel: 'Ad',
    cloneFrom: 'Şuradan klonla',
    cloneFromNone: 'Hiçbiri (boş)',
    cloneFromDesc: "Seçilen kaynak profilden yapılandırmayı, becerileri ve SOUL.md'yi kopyalar.",
    cloneFromDefault: 'Varsayılandan klonla',
    cloneFromDefaultDesc: "Varsayılan profilden yapılandırma, beceriler ve SOUL.md'yi kopyalar.",
    invalidName: hint => `Geçersiz ad. ${hint}`,
    nameRequired: 'Ad gereklidir.',
    creating: 'Oluşturuluyor...',
    createAction: 'Profil oluştur',
    renameTitle: 'Profili yeniden adlandır',
    renameDescPrefix: 'Yeniden adlandırma, profil dizinini ve ',
    renameDescSuffix: ' içindeki sarmalayıcı komut dosyalarını günceller.',
    newNameLabel: 'Yeni ad',
    renaming: 'Yeniden adlandırılıyor...',
    created: 'Profil oluşturuldu',
    renamed: 'Profil yeniden adlandırıldı',
    deleted: 'Profil silindi',
    setupCopied: 'Kurulum komutu kopyalandı',
    soulSaved: 'SOUL.md kaydedildi',
    failedLoad: 'Profiller yüklenemedi',
    failedDelete: 'Profil silinemedi',
    failedCopy: 'Kurulum komutu kopyalanamadı',
    failedLoadSoul: 'SOUL.md yüklenemedi',
    failedSaveSoul: 'SOUL.md kaydedilemedi',
    failedCreate: 'Profil oluşturulamadı',
    failedRename: 'Profil yeniden adlandırılamadı'
  },

  cron: {
    close: "Cron'u kapat",
    search: 'Cron işlerinde ara...',
    loading: 'Cron işleri yükleniyor...',
    states: {
      enabled: 'etkin',
      scheduled: 'zamanlandı',
      running: 'çalışıyor',
      paused: 'duraklatıldı',
      disabled: 'devre dışı',
      error: 'hata',
      completed: 'tamamlandı'
    },
    deliveryLabels: {
      local: 'Bu masaüstü',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-posta'
    },
    scheduleLabels: {
      daily: 'Günlük',
      weekdays: 'Hafta içi',
      weekly: 'Haftalık',
      monthly: 'Aylık',
      hourly: 'Saatlik',
      'every-15-minutes': 'Her 15 dakikada',
      custom: 'Özel'
    },
    scheduleHints: {
      daily: 'Her gün saat 09:00',
      weekdays: "Pazartesiden Cumaya saat 09:00'da",
      weekly: "Her Pazartesi saat 09:00'da",
      monthly: "Her ayın 1'inde saat 09:00'da",
      hourly: 'Her saatin başında',
      'every-15-minutes': 'Her 15 dakikada bir',
      custom: 'Cron sözdizimi veya doğal dil'
    },
    days: {
      '0': 'Pazar',
      '1': 'Pazartesi',
      '2': 'Salı',
      '3': 'Çarşamba',
      '4': 'Perşembe',
      '5': 'Cuma',
      '6': 'Cumartesi',
      '7': 'Pazar'
    },
    dayFallback: value => `${value}. gün`,
    everyDayAt: time => `Her gün saat ${time}`,
    weekdaysAt: time => `Hafta içi saat ${time}`,
    everyDayOfWeekAt: (day, time) => `Her ${day} saat ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Her ayın ${dayOfMonth}. günü saat ${time}`,
    topOfHour: 'Her saatin başında',
    everyHourAt: minute => `Her saat :${minute}`,
    newCron: 'Yeni cron',
    emptyDescNew:
      "Bir cron ifadesiyle çalışacak bir istem zamanlayın. Hermes çalıştıracak ve sonuçları seçtiğiniz hedefe iletecektir.",
    emptyDescSearch: 'Daha geniş bir arama sorgusu deneyin.',
    emptyTitleNew: 'Henüz zamanlanmış iş yok',
    emptyTitleSearch: 'Eşleşme yok',
    last: 'Son:',
    next: 'Sonraki:',
    noRuns: 'Henüz çalışma yok',
    manage: 'Yönet',
    showRuns: 'Çalışmaları göster',
    hideRuns: 'Çalışmaları gizle',
    runHistory: 'Çalışma geçmişi',
    actionsFor: title => `${title} için eylemler`,
    actionsTitle: 'Cron iş eylemleri',
    resume: "Cron'u sürdür",
    pause: "Cron'u duraklat",
    resumeTitle: 'Sürdür',
    pauseTitle: 'Duraklat',
    triggerNow: 'Şimdi tetikle',
    edit: "Cron'u düzenle",
    deleteTitle: 'Cron işi silinsin mi?',
    deleteDescPrefix: 'Bu işlem ',
    deleteDescSuffix: ' öğesini kalıcı olarak kaldırır. Hemen duracaktır.',
    deleting: 'Siliniyor...',
    resumed: 'Cron sürdürüldü',
    paused: 'Cron duraklatıldı',
    triggered: 'Cron tetiklendi',
    deleted: 'Cron silindi',
    created: 'Cron oluşturuldu',
    updated: 'Cron güncellendi',
    failedLoad: 'Cron işleri yüklenemedi',
    failedUpdate: 'Cron işi güncellenemedi',
    failedTrigger: 'Cron işi tetiklenemedi',
    failedDelete: 'Cron işi silinemedi',
    failedSave: 'Cron işi kaydedilemedi',
    editTitle: 'Cron işini düzenle',
    createTitle: 'Yeni cron işi',
    editDesc: 'Zamanlamayı, istemi veya teslim hedefini güncelleyin. Değişiklikler bir sonraki çalıştırmada uygulanır.',
    createDesc:
      'Otomatik çalışacak bir istem zamanlayın. Cron sözdizimi veya "her 15 dakikada" gibi doğal ifadeler kullanın.',
    nameLabel: 'Ad',
    namePlaceholder: 'Sabah özeti',
    promptLabel: 'İstem',
    promptPlaceholder: 'Her çalıştırmada ajan ne yapacak?',
    frequencyLabel: 'Sıklık',
    deliverLabel: 'Teslim yeri',
    customScheduleLabel: 'Özel zamanlama',
    customPlaceholder: "0 9 * * * veya hafta içi saat 9'da",
    customHint: "Cron ifadesi ya da \"her saat\" veya \"hafta içi saat 9'da\" gibi ifadeler.",
    optional: 'İsteğe bağlı',
    promptScheduleRequired: 'İstem ve zamanlama gereklidir.',
    saveChanges: 'Değişiklikleri kaydet',
    createAction: 'Cron oluştur'
  },

  artifacts: {
    search: 'Yapıtlarda ara...',
    refresh: 'Yapıtları yenile',
    refreshing: 'Yapıtlar yenileniyor',
    indexing: 'Son oturum yapıtları dizine ekleniyor',
    tabAll: 'Tümü',
    tabImages: 'Görseller',
    tabFiles: 'Dosyalar',
    tabLinks: 'Bağlantılar',
    noArtifactsTitle: 'Yapıt bulunamadı',
    noArtifactsDesc: 'Oturumların ürettiği görseller ve dosya çıktıları burada görünür.',
    failedLoad: 'Yapıtlar yüklenemedi',
    openFailed: 'Açma başarısız',
    itemsImage: 'görsel',
    itemsLink: 'bağlantı',
    itemsFile: 'dosya',
    itemsGeneric: 'öğe',
    zero: '0',
    rangeOf: (start, end, total) => `${total} öğeden ${start}-${end}`,
    goToPage: (itemLabel, page) => `${itemLabel} sayfa ${page}'e git`,
    colTitleLink: 'Bağlantı başlığı',
    colTitleFile: 'Ad',
    colTitleDefault: 'Başlık / ad',
    colLocationLink: 'URL',
    colLocationFile: 'Yol',
    colLocationDefault: 'Konum',
    colSession: 'Oturum',
    kindImage: 'görsel',
    kindFile: 'dosya',
    kindLink: 'bağlantı',
    chat: 'Sohbet',
    copyUrl: "URL'yi kopyala",
    copyPath: 'Yolu kopyala'
  },

  sidebar: {
    nav: {
      'new-session': 'Yeni oturum',
      skills: 'Beceriler ve Araçlar',
      messaging: 'Mesajlaşma',
      artifacts: 'Yapıtlar'
    },
    searchAria: 'Oturumları ara',
    searchPlaceholder: 'Oturumlarda ara…',
    clearSearch: 'Aramayı temizle',
    noMatch: query => `"${query}" ile eşleşen oturum yok.`,
    results: 'Sonuçlar',
    pinned: 'Sabitlenmiş',
    sessions: 'Oturumlar',
    cronJobs: 'Cron işleri',
    groupAriaGrouped: 'Oturumları tek liste olarak göster',
    groupAriaUngrouped: 'Oturumları çalışma alanına göre grupla',
    groupTitleGrouped: "Oturumların grubunu kaldır",
    groupTitleUngrouped: 'Çalışma alanına göre grupla',
    allPinned:
      'Buradaki her şey sabitlenmiş. Son sohbetlerde görmek için bir sohbetin sabitlemesini kaldırın.',
    shiftClickHint: 'Sabitlemek için Shift+tıkla',
    noWorkspace: 'Çalışma alanı yok',
    newSessionIn: label => `${label} içinde yeni oturum`,
    reorderWorkspace: label => `${label} çalışma alanını yeniden sırala`,
    showMoreIn: (count, label) => `${label} içinde ${count} tane daha göster`,
    loading: 'Yükleniyor…',
    loadMore: 'Daha fazla yükle',
    loadCount: step => `${step} tane daha yükle`,
    row: {
      pin: 'Sabitle',
      unpin: 'Sabitlemeden çıkar',
      copyId: 'Kimliği kopyala',
      export: 'Dışa aktar',
      rename: 'Yeniden adlandır',
      archive: 'Arşivle',
      newWindow: 'Yeni pencere',
      copyIdFailed: 'Oturum kimliği kopyalanamadı',
      actionsFor: title => `${title} için eylemler`,
      sessionActions: 'Oturum eylemleri',
      sessionRunning: 'Oturum çalışıyor',
      needsInput: 'Girdiniz gerekiyor',
      waitingForAnswer: 'Yanıtınız bekleniyor',
      handoffOrigin: platform => `${platform} üzerinden devralındı`,
      renamed: 'Yeniden adlandırıldı',
      renameFailed: 'Yeniden adlandırma başarısız',
      renameTitle: 'Oturumu yeniden adlandır',
      renameDesc: 'Bu sohbete akılda kalıcı bir başlık verin. Temizlemek için boş bırakın.',
      untitledPlaceholder: 'Adsız oturum',
      ageNow: 'şimdi',
      ageDay: 'g',
      ageHour: 's',
      ageMin: 'dk'
    }
  },

  composer: {
    message: 'Mesaj',
    wakingProfile: profile => `${profile} uyandırılıyor…`,
    placeholderStarting: 'Hermes başlatılıyor...',
    placeholderReconnecting: "Hermes'e yeniden bağlanılıyor…",
    placeholderFollowUp: 'Takip gönderin',
    newSessionPlaceholders: [
      'Ne inşa ediyoruz?',
      "Hermes'e bir görev verin",
      'Aklınızda ne var?',
      'İhtiyacınızı açıklayın',
      'Neyle uğraşalım?',
      'Her şeyi sorun',
      'Bir hedefle başlayın'
    ],
    followUpPlaceholders: [
      'Takip gönderin',
      'Daha fazla bağlam ekleyin',
      'İsteği hassaslaştırın',
      'Sırada ne var?',
      'Devam edin',
      'Daha da ilerleyin',
      'Ayarlayın veya sürdürün'
    ],
    startVoice: 'Sesli konuşma başlat',
    queueMessage: 'Mesajı sıraya koy',
    steer: 'Mevcut çalışmayı yönlendir',
    stop: 'Durdur',
    send: 'Gönder',
    speaking: 'Konuşuyor',
    transcribing: 'Döküm yapılıyor',
    thinking: 'Düşünüyor',
    muted: 'Sessiz',
    listening: 'Dinliyor',
    muteMic: 'Mikrofonu kapat',
    unmuteMic: 'Mikrofonu aç',
    stopListening: 'Dinlemeyi durdur ve gönder',
    stopShort: 'Durdur',
    endConversation: 'Sesli konuşmayı bitir',
    endShort: 'Bitir',
    stopDictation: 'Dikte etmeyi durdur',
    transcribingDictation: 'Dikte dökümü yapılıyor',
    voiceDictation: 'Sesli dikte',
    lookupLoading: 'Aranıyor…',
    lookupNoMatches: 'Eşleşme yok.',
    lookupTry: 'Dene',
    lookupOr: 'veya',
    commonCommands: 'Yaygın komutlar',
    hotkeys: 'Kısayol tuşları',
    helpFooter: 'tam paneli açar · geri al ile kapat',
    commandDescs: {
      '/help': 'komutların ve kısayolların tam listesi',
      '/clear': 'yeni oturum başlat',
      '/resume': 'önceki oturumu sürdür',
      '/details': 'transkript ayrıntı düzeyini kontrol et',
      '/copy': 'seçimi veya son asistan mesajını kopyala',
      '/quit': "hermes'ten çık"
    },
    hotkeyDescs: {
      'composer.mention': "dosyaları, klasörleri, URL'leri, git'i referansla",
      'composer.slash': 'eğik çizgi komut paleti',
      'composer.help': 'bu hızlı yardım (kapatmak için sil)',
      'composer.sendNewline': 'gönder · yeni satır için Shift+Enter',
      'composer.sendQueued': 'sıradaki turu gönder',
      'keybinds.openPanel': 'tüm klavye kısayolları',
      'composer.cancel': 'açılır pencereyi kapat · çalışmayı iptal et',
      'composer.history': 'açılır pencere / geçmişi döngüle'
    },
    attachUrlTitle: 'URL ekle',
    attachUrlDesc: 'Hermes sayfayı getirecek ve bu turun bağlamı olarak ekleyecektir.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Tam URL\'yi ekleyin, örn. ',
    attach: 'Ekle',
    queued: count => `${count} Sırada`,
    attachmentOnly: 'Yalnızca ek turu',
    emptyTurn: 'Boş tur',
    attachments: count => `${count} ek`,
    editingInComposer: 'Düzenleyicide düzenleniyor',
    editingQueuedInComposer: 'Düzenleyicide sıradaki tur düzenleniyor',
    queueEdit: 'Düzenle',
    queueSendNext: 'Sonraki',
    queueSend: 'Gönder',
    queueDelete: 'Sil',
    queueStuckTitle: 'Sıradaki mesaj gönderilemedi',
    queueStuckBody:
      'Sıradaki bir tur tekrar tekrar gönderilmekte başarısız oldu. Hâlâ sırada — tekrar göndermeyi deneyin.',
    previewUnavailable: 'Önizleme kullanılamıyor',
    previewLabel: label => `${label} önizlemesi`,
    couldNotPreview: label => `${label} önizlenemedi`,
    removeAttachment: label => `${label} kaldır`,
    dictating: 'Dikte ediliyor',
    preparingAudio: 'Ses hazırlanıyor',
    speakingResponse: 'Yanıt okunuyor',
    readingAloud: 'Sesli okunuyor',
    themeSuggestions: 'Masaüstü tema önerileri',
    noMatchingThemes: 'Eşleşen tema yok.',
    themeTryPre: 'Deneyin: ',
    themeTryPost: '.',
    attachLabel: 'Ekle',
    files: 'Dosyalar…',
    folder: 'Klasör…',
    images: 'Görseller…',
    pasteImage: 'Görsel yapıştır',
    url: 'URL…',
    promptSnippets: 'İstem parçacıkları…',
    tipPre: 'İpucu: dosyaları satır içi referanslamak için ',
    tipPost: ' yazın.',
    snippetsTitle: 'İstem parçacıkları',
    snippetsDesc: 'Düzenleyiciye eklemek için bir başlangıç istemi seçin.',
    dropFiles: 'Eklemek için dosya bırakın',
    dropSession: 'Bu sohbeti bağlamak için bırakın',
    snippets: {
      codeReview: {
        label: 'Kod incelemesi',
        description:
          'Mevcut değişikliği gerilemeler, atlanmış uç durumlar ve eksik testler açısından denetleyin.',
        text: 'Lütfen bunu hatalar, gerilemeler ve eksik testler açısından inceleyin.'
      },
      implementationPlan: {
        label: 'Uygulama planı',
        description: 'Farka odaklanmak için koda dokunmadan önce bir yaklaşım özetleyin.',
        text: 'Lütfen kodu değiştirmeden önce kısa bir uygulama planı yapın.'
      },
      explainThis: {
        label: 'Bunu açıkla',
        description: 'Seçili kodun nasıl çalıştığını anlatın ve temel dosyalara bağlantı verin.',
        text: 'Lütfen bunun nasıl çalıştığını açıklayın ve temel dosyaları gösterin.'
      }
    }
  },

  statusStack: {
    agents: 'Ajanlar',
    background: count => `${count} Arka planda`,
    subagents: count => `${count} Alt ajan`,
    todos: (done, total) => `Görevler ${done}/${total}`,
    running: 'Çalışıyor',
    stop: 'Durdur',
    dismiss: 'Kapat',
    exit: code => `çıkış ${code}`
  },

  updates: {
    stages: {
      idle: 'Hazırlanıyor…',
      prepare: 'Hazırlanıyor…',
      fetch: 'İndiriliyor…',
      pull: 'Neredeyse bitti…',
      pydeps: 'Tamamlanıyor…',
      restart: 'Hermes yeniden başlatılıyor…',
      manual: 'Terminalinizden güncelleyin',
      error: 'Güncelleme duraklatıldı'
    },
    checking: 'Güncellemeler aranıyor…',
    checkFailedTitle: 'Güncellemeler kontrol edilemedi',
    tryAgain: 'Yeniden dene',
    notAvailableTitle: 'Güncelleme mevcut değil',
    unsupportedMessage: "Hermes'in bu sürümü kendini uygulama içinden güncelleyemiyor.",
    connectionRetry: 'Bağlantınızı kontrol edin ve yeniden deneyin.',
    latestBody: 'En son sürümü çalıştırıyorsunuz.',
    latestBodyBackend: 'Arka uç en son sürümü çalıştırıyor.',
    allSetTitle: 'Hazırsınız',
    availableTitle: 'Yeni güncelleme mevcut',
    availableBody: "Hermes'in yeni bir sürümü kurulmaya hazır.",
    availableTitleBackend: 'Arka uç güncellemesi mevcut',
    availableBodyBackend: 'Bağlı Hermes arka ucunun daha yeni bir sürümü kurulmaya hazır.',
    availableBodyNoChangelog:
      'Daha yeni bir sürüm hazır. Bu kurulum türü için sürüm notları mevcut değil.',
    updateNow: 'Şimdi güncelle',
    maybeLater: 'Belki daha sonra',
    moreChanges: count => `+ ${count} değişiklik daha dahil.`,
    manualTitle: 'Terminalinizden güncelleyin',
    manualBody:
      "Hermes'i komut satırından yüklediniz, bu nedenle güncellemeler de orada çalışır. Bunu terminalinize yapıştırın:",
    manualPickedUp: "Hermes bir sonraki başlatmada yeni sürümü alacaktır.",
    copy: 'Kopyala',
    copied: 'Kopyalandı',
    done: 'Tamam',
    applyingBody:
      "Hermes güncelleyicisi kendi penceresinde devralacak ve işi bitince Hermes'i yeniden açacaktır.",
    applyingBodyBackend:
      "Uzak arka uç güncellemeyi uygulayıp yeniden başlayacak. Geri geldiğinde Hermes otomatik olarak yeniden bağlanır.",
    applyingClose: 'Hermes güncellemeyi uygulamak için kapanacaktır.',
    errorTitle: 'Güncelleme tamamlanamadı',
    errorBody: 'Endişelenmeyin — hiçbir şey kaybolmadı. Şimdi yeniden deneyebilirsiniz.',
    notNow: 'Şimdi değil',
    applyStatus: {
      preparing: 'Arka uç güncelleniyor…',
      pulling: 'Arka uç güncelleniyor…',
      restarting: 'Arka uç güncellemeyi yüklemek için yeniden başlatılıyor…',
      notAvailable: 'Bu arka uç için güncelleme mevcut değil.',
      failed: 'Arka uç güncellemesi başarısız.',
      noReturn:
        'Arka uç çevrimiçi dönmedi. Güncelleme tamamlanmamış olabilir — arka uç ana bilgisayarını kontrol edin.'
    }
  },

  install: {
    stageStates: {
      pending: 'Bekliyor',
      running: 'Yükleniyor',
      succeeded: 'Tamamlandı',
      skipped: 'Atlandı',
      failed: 'Başarısız'
    },
    oneTimeTitle: 'Hermes tek seferlik kurulum gerektiriyor',
    unsupportedDesc: platform =>
      `${platform} üzerinde otomatik ilk başlatma kurulumu henüz mevcut değil. Terminal'i açın ve aşağıdaki komutu çalıştırın, ardından bu uygulamayı yeniden başlatın. Sonraki başlatmalarda bu adım atlanacaktır.`,
    installCommand: 'Kurulum komutu',
    copyCommand: 'Komutu kopyala',
    viewDocs: 'Kurulum belgelerini görüntüle',
    installTo: 'Kurulacak yer',
    retryAfterRun: 'Çalıştırdım — yeniden dene',
    failedTitle: 'Kurulum başarısız',
    settingUpTitle: 'Hermes Agent kuruluyor',
    finishingTitle: 'Tamamlanıyor',
    failedDesc:
      "Kurulum adımlarından biri başarısız oldu. Windows'ta bu, başka bir Hermes CLI veya masaüstü örneği çalışıyorsa olabilir. Çalışan Hermes örneklerini durdurun, ardından yeniden deneyin. Aşağıdaki ayrıntılara veya masaüstü günlüğüne bakın.",
    activeDesc:
      'Bu tek seferlik bir kurulumdur. Hermes yükleyicisi bağımlılıkları indirip makinenizi yapılandırıyor. Sonraki başlatmalarda bu adım atlanacaktır.',
    progress: (completed, total) => `${total} adımın ${completed} tanesi tamamlandı`,
    currentStage: stage => ` — şimdi: ${stage}`,
    fetchingManifest: 'Yükleyici manifestosu alınıyor...',
    error: 'Hata',
    hideOutput: 'Yükleyici çıktısını gizle',
    showOutput: 'Yükleyici çıktısını göster',
    lines: count => `${count} satır`,
    noOutput: 'Henüz çıktı yok.',
    cancelling: 'İptal ediliyor...',
    cancelInstall: 'Kurulumu iptal et',
    transcriptSaved: 'Tam transkript kaydedildi:',
    copiedOutput: 'Kopyalandı!',
    copyOutput: 'Çıktıyı kopyala',
    reloadRetry: 'Yeniden yükle ve dene'
  },

  onboarding: {
    headerTitle: "Hermes Agent'ı kuralım",
    headerDesc: 'Sohbete başlamak için bir model sağlayıcısı bağlayın. Çoğu seçenek tek tıklamayla olur.',
    preparingInstall:
      'Hermes kurulumu tamamlıyor. Bu genellikle ilk çalıştırmada bir dakikadan az sürer.',
    starting: 'Hermes başlatılıyor…',
    lookingUpProviders: 'Sağlayıcılar aranıyor...',
    collapse: 'Daralt',
    otherProviders: 'Diğer sağlayıcılar',
    haveApiKey: 'Bir API anahtarım var',
    chooseLater: 'Sağlayıcıyı sonra seçeceğim',
    recommended: 'Önerilen',
    connected: 'Bağlandı',
    featuredPitch: "Tek abonelik, 300'den fazla öncü model — Hermes'i çalıştırmanın önerilen yolu",
    openRouterPitch: 'Tek anahtar, yüzlerce model — sağlam bir varsayılan',
    apiKeyOptions: {
      openrouter: {
        short: 'tek anahtar, pek çok model',
        description:
          'Tek bir anahtar arkasında yüzlerce modeli barındırır. Yeni kurulumlar için iyi bir varsayılan.'
      },
      openai: { short: 'GPT sınıfı modeller', description: 'OpenAI modellerine doğrudan erişim.' },
      gemini: {
        short: 'Gemini modelleri',
        description: 'Google Gemini modellerine doğrudan erişim.'
      },
      xai: { short: 'Grok modelleri', description: 'xAI Grok modellerine doğrudan erişim.' },
      local: {
        short: 'kendi barındırma',
        description:
          "Hermes'i yerel veya kendi barındırdığınız OpenAI uyumlu bir uç noktaya yönlendirin (vLLM, llama.cpp, Ollama vb.)."
      }
    },
    backToSignIn: 'Oturum açmaya geri dön',
    getKey: 'Anahtar al',
    replaceCurrent: 'Mevcut değeri değiştir',
    pasteApiKey: 'API anahtarını yapıştır',
    localApiKeyPlaceholder: 'API anahtarı (isteğe bağlı — yalnızca uç noktanız gerektiriyorsa)',
    couldNotSave: 'Kimlik bilgisi kaydedilemedi.',
    connecting: 'Bağlanıyor',
    update: 'Güncelle',
    flowSubtitles: {
      pkce: 'Oturum açmak için tarayıcınızı açar, ardından buraya döner',
      device_code: "Tarayıcınızda doğrulama sayfası açar — Hermes otomatik bağlanır",
      loopback: "Oturum açmak için tarayıcınızı açar — Hermes otomatik bağlanır",
      external: 'Terminalinizde bir kez oturum açın, ardından sohbete dönün'
    },
    startingSignIn: provider => `${provider} için oturum açma başlatılıyor...`,
    verifyingCode: provider => `${provider} ile kodunuz doğrulanıyor...`,
    connectedProvider: provider => `${provider} bağlandı`,
    connectedPicking: provider => `${provider} bağlandı. Varsayılan model seçiliyor...`,
    signInFailed: 'Oturum açma başarısız. Yeniden deneyin.',
    pickDifferentProvider: 'Farklı bir sağlayıcı seçin',
    signInWith: provider => `${provider} ile oturum aç`,
    openedBrowser: provider => `${provider} tarayıcınızda açıldı.`,
    authorizeThere: "Hermes'i orada yetkilendirin.",
    copyAuthCode: 'Yetkilendirme kodunu kopyalayın ve aşağıya yapıştırın.',
    pasteAuthCode: 'Yetkilendirme kodunu yapıştır',
    reopenAuthPage: 'Yetkilendirme sayfasını yeniden aç',
    autoBrowser: provider =>
      `${provider} tarayıcınızda açıldı. Hermes'i orada yetkilendirin, otomatik olarak bağlanacaksınız — kopyalama veya yapıştırma gerekmez.`,
    reopenSignInPage: 'Oturum açma sayfasını yeniden aç',
    waitingAuthorize: 'Yetkilendirmeniz bekleniyor...',
    externalPending: provider =>
      `${provider} kendi CLI'ı üzerinden oturum açar. Bu komutu bir terminalde çalıştırın, ardından geri dönüp "Oturum açtım" seçeneğini belirleyin:`,
    signedIn: 'Oturum açtım',
    deviceCodeOpened: provider => `${provider} tarayıcınızda açıldı. Bu kodu oraya girin:`,
    reopenVerification: 'Doğrulama sayfasını yeniden aç',
    copy: 'Kopyala',
    defaultModel: 'Varsayılan model',
    freeTier: 'Ücretsiz katman',
    pro: 'Pro',
    free: 'Ücretsiz',
    price: (input, output) => `${input} giriş / ${output} çıkış / Mtok`,
    change: 'Değiştir',
    startChatting: 'Başla',
    docs: provider => `${provider} belgeleri`
  },

  modelPicker: {
    title: 'Model değiştir',
    current: 'mevcut:',
    unknown: '(bilinmiyor)',
    search: 'Sağlayıcıları ve modelleri filtrele...',
    noModels: 'Model bulunamadı.',
    persistGlobalSession: 'Genel olarak kalıcı yap (aksi takdirde yalnızca bu oturum)',
    persistGlobal: 'Genel olarak kalıcı yap',
    addProvider: 'Sağlayıcı ekle',
    loadFailed: 'Modeller yüklenemedi',
    noAuthenticatedProviders: 'Kimliği doğrulanmış sağlayıcı yok.',
    pro: 'Pro',
    proNeedsSubscription: 'Pro modeller ücretli Nous aboneliği gerektirir.',
    free: 'Ücretsiz',
    freeTier: 'Ücretsiz katman',
    priceTitle: 'Milyon jeton başına giriş / çıkış fiyatı'
  },

  modelVisibility: {
    title: 'Modeller',
    search: 'Model ara',
    noAuthenticatedProviders: 'Kimliği doğrulanmış sağlayıcı yok.',
    addProvider: 'Sağlayıcı ekle…'
  },

  shell: {
    windowControls: 'Pencere denetimleri',
    paneControls: 'Bölme denetimleri',
    appControls: 'Uygulama denetimleri',
    modelMenu: {
      search: 'Model ara',
      noModels: 'Model bulunamadı',
      editModels: 'Modelleri düzenle…',
      fast: 'Hızlı',
      medium: 'Orta'
    },
    modelOptions: {
      noOptions: 'Bu model için seçenek yok',
      options: 'Seçenekler',
      thinking: 'Düşünme',
      fast: 'Hızlı',
      effort: 'Çaba',
      minimal: 'Minimal',
      low: 'Düşük',
      medium: 'Orta',
      high: 'Yüksek',
      max: 'Maksimum',
      updateFailed: 'Model seçeneği güncellemesi başarısız',
      fastFailed: 'Hızlı mod güncellemesi başarısız'
    },
    gatewayMenu: {
      gateway: 'Ağ geçidi',
      connected: 'Bağlandı',
      connecting: 'Bağlanıyor',
      offline: 'Çevrimdışı',
      inferenceReady: 'Çıkarım hazır',
      inferenceNotReady: 'Çıkarım hazır değil',
      checkingInference: 'Çıkarım kontrol ediliyor',
      disconnected: 'Bağlantı kesildi',
      openSystem: 'Sistem panelini aç',
      connection: label => `Bağlantı: ${label}`,
      recentActivity: 'Son etkinlik',
      viewAllLogs: 'Tüm günlükleri görüntüle →',
      messagingPlatforms: 'Mesajlaşma platformları'
    },
    statusbar: {
      unknown: 'bilinmiyor',
      restart: 'yeniden başlat',
      update: 'güncelle',
      updateInProgress: 'Güncelleme devam ediyor',
      commitsBehind: (count, branch) => `${branch} dalının ${count} commit gerisinde`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `Arka uç v${version}`,
      clientLabel: version => `istemci v${version}`,
      backendLabel: version => `arka uç v${version}`,
      commit: sha => `commit ${sha}`,
      branch: branch => `dal ${branch}`,
      closeCommandCenter: 'Komut Merkezini kapat',
      openCommandCenter: 'Komut Merkezini aç',
      showTerminal: 'Terminali göster',
      hideTerminal: 'Terminali gizle',
      gateway: 'Ağ geçidi',
      gatewayReady: 'hazır',
      gatewayNeedsSetup: 'kurulum gerekli',
      gatewayChecking: 'kontrol ediliyor',
      gatewayConnecting: 'bağlanıyor',
      gatewayOffline: 'çevrimdışı',
      gatewayTitle: 'Hermes çıkarım ağ geçidi durumu',
      agents: 'Ajanlar',
      closeAgents: 'Ajanları kapat',
      openAgents: 'Ajanları aç',
      subagents: count => `${count} alt ajan`,
      failed: count => `${count} başarısız`,
      running: count => `${count} çalışıyor`,
      cron: 'Cron',
      openCron: 'Cron işlerini aç',
      turnRunning: 'Çalışıyor',
      currentTurnElapsed: 'Mevcut tur süresi',
      contextUsage: 'Bağlam kullanımı',
      session: 'Oturum',
      runtimeSessionElapsed: 'Çalışma zamanı oturum süresi',
      yoloOn:
        'YOLO açık — tehlikeli komutlar otomatik onaylanıyor. Kapatmak için tıklayın. Genel geçiş için Shift+tıklayın.',
      yoloOff:
        'YOLO kapalı — tehlikeli komutları otomatik onaylamak için tıklayın. Genel geçiş için Shift+tıklayın.',
      modelNone: 'yok',
      noModel: 'model yok',
      switchModel: 'Model değiştir',
      openModelPicker: 'Model seçiciyi aç',
      modelTitle: (provider, model) => `Model · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },

  rightSidebar: {
    aria: 'Sağ kenar çubuğu',
    panelsAria: 'Sağ kenar çubuğu panelleri',
    files: 'Dosya sistemi',
    terminal: 'Terminal',
    noFolderSelected: 'Klasör seçilmedi',
    changeCwdTitle: 'Çalışma dizinini değiştir',
    remotePickerTitle: 'Uzak klasör seç',
    remotePickerDescription: 'Bağlı arka uçtaki klasörlere göz atın.',
    remotePickerSelect: 'Klasörü seç',
    folderTip: cwd => `${cwd} — klasörü değiştirmek için tıklayın`,
    openFolder: 'Klasörü aç',
    refreshTree: 'Ağacı yenile',
    collapseAll: 'Tüm klasörleri daralt',
    previewUnavailable: 'Önizleme kullanılamıyor',
    couldNotPreview: path => `${path} önizlenemedi`,
    noProjectTitle: 'Proje yok',
    noProjectBody: 'Dosyalara göz atmak için durum çubuğundan bir çalışma dizini ayarlayın.',
    unreadableTitle: 'Okunamıyor',
    unreadableBody: error => `Bu klasör okunamadı (${error}).`,
    emptyTitle: 'Boş',
    emptyBody: 'Bu klasör boş.',
    treeErrorTitle: 'Ağaç hatası',
    treeErrorBody: 'Dosya ağacı bu klasörü görüntülerken hatayla karşılaştı.',
    tryAgain: 'Yeniden dene',
    loadingTree: 'Dosya ağacı yükleniyor',
    loadingFiles: 'Dosyalar yükleniyor',
    terminalHide: 'Terminali gizle',
    addToChat: 'Sohbete ekle'
  },

  preview: {
    tab: 'Önizleme',
    closeTab: label => `${label} kapat`,
    closePane: 'Önizleme bölmesini kapat',
    loading: 'Önizleme yükleniyor',
    unavailable: 'Önizleme kullanılamıyor',
    opening: 'Açılıyor...',
    hide: 'Gizle',
    openPreview: 'Önizlemeyi aç',
    sourceLineTitle: 'Seçmek için tıkla · genişletmek için Shift+tıkla · düzenleyiciye sürükle',
    source: 'KAYNAK',
    renderedPreview: 'ÖNİZLEME',
    unknownSize: 'bilinmeyen boyut',
    binaryTitle: 'Bu bir ikili dosyaya benziyor',
    binaryBody: label => `${label} önizlemesi okunamaz metin gösterebilir.`,
    largeTitle: 'Bu dosya büyük',
    largeBody: (label, size) => `${label}, ${size} boyutunda. Hermes yalnızca ilk 512 KB'ı gösterir.`,
    previewAnyway: 'Yine de önizle',
    truncated: 'İlk 512 KB gösteriliyor.',
    noInlineTitle: 'Satır içi önizleme yok',
    noInlineBody: mimeType => `${mimeType || 'Bu dosya türü'} yine de bağlam olarak eklenebilir.`,
    console: {
      deselect: 'Girdinin seçimini kaldır',
      select: 'Girdiyi seç',
      copyFailed: 'Konsol çıktısı kopyalanamadı',
      copyEntry: 'Bu girdiyi kopyala',
      sendEntry: 'Bu girdiyi sohbete gönder',
      messages: count => `${count} konsol mesajı`,
      resize: 'Önizleme konsolunu yeniden boyutlandır',
      title: 'Önizleme Konsolu',
      selected: count => `${count} seçildi`,
      sendToChat: 'Sohbete gönder',
      copySelected: 'Seçileni panoya kopyala',
      copyAll: 'Tümünü panoya kopyala',
      copy: 'Kopyala',
      clear: 'Temizle',
      empty: 'Henüz konsol mesajı yok.',
      promptHeader: 'Önizleme konsolu:',
      sentTitle: 'Sohbete gönderildi',
      sentMessage: count => `${count} günlük girdisi düzenleyiciye eklendi`
    },
    web: {
      appFailedToBoot: 'Önizleme uygulaması başlatılamadı',
      serverNotFound: 'Sunucu bulunamadı',
      failedToLoad: 'Önizleme yüklenemedi',
      tryAgain: 'Yeniden dene',
      restarting: 'Hermes yeniden başlatılıyor...',
      askRestart: "Hermes'ten sunucuyu yeniden başlatmasını iste",
      lookingRestart: taskId =>
        `Hermes yeniden başlatılacak önizleme sunucusunu arıyor (${taskId})`,
      restartingTitle: 'Önizleme sunucusu yeniden başlatılıyor',
      restartingMessage:
        'Hermes arka planda çalışıyor. İlerleme için önizleme konsolunu izleyin.',
      startRestartFailed: message => `Sunucu yeniden başlatması başlatılamadı: ${message}`,
      restartFailed: 'Sunucu yeniden başlatması başarısız',
      hideConsole: 'Önizleme konsolunu gizle',
      showConsole: 'Önizleme konsolunu göster',
      hideDevTools: "Önizleme DevTools'u gizle",
      openDevTools: "Önizleme DevTools'unu aç",
      finishedRestarting: message =>
        `Hermes önizleme sunucusunu yeniden başlatmayı tamamladı${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Sunucu yeniden başlatması başarısız: ${message}`,
      unknownError: 'bilinmeyen hata',
      restartedTitle: 'Önizleme sunucusu yeniden başlatıldı',
      reloadingNow: 'Önizleme şimdi yeniden yükleniyor.',
      restartFailedTitle: 'Önizleme yeniden başlatması başarısız',
      restartFailedMessage: 'Hermes sunucuyu yeniden başlatamadı.',
      stillWorking:
        'Hermes hâlâ çalışıyor ancak henüz yeniden başlatma sonucu gelmedi. Sunucu komutu ön planda çalışıyor olabilir.',
      workspaceReloading: 'Çalışma alanı değişti, önizleme yeniden yükleniyor',
      fileChanged: url => `Dosya değişti, önizleme yeniden yükleniyor: ${url}`,
      filesChanged: (count, url) => `${count} dosya değişikliği, önizleme yeniden yükleniyor: ${url}`,
      watchFailed: message => `Önizleme dosyası izlenemedi: ${message}`,
      moduleMimeDescription:
        "Modül komut dosyaları yanlış MIME türüyle sunuluyor. Bu genellikle proje geliştirme sunucusu yerine statik dosya sunucusunun Vite/React uygulamasını sunduğu anlamına gelir.",
      loadFailedConsole: (code, message) =>
        `Yükleme başarısız${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'Önizleme sayfasına ulaşılamadı.',
      openTarget: url => `${url} aç`,
      fallbackTitle: 'Önizleme'
    }
  },

  assistant: {
    thread: {
      loadingSession: 'Oturum yükleniyor',
      showEarlier: 'Önceki mesajları göster',
      loadingResponse: 'Hermes bir yanıt yüklüyor',
      thinking: 'Düşünüyor',
      today: time => `Bugün, ${time}`,
      yesterday: time => `Dün, ${time}`,
      copy: 'Kopyala',
      refresh: 'Yenile',
      moreActions: 'Daha fazla eylem',
      branchNewChat: 'Yeni sohbette dallan',
      readAloudFailed: 'Sesli okuma başarısız',
      preparingAudio: 'Ses hazırlanıyor...',
      stopReading: 'Okumayı durdur',
      readAloud: 'Sesli oku',
      editMessage: 'Mesajı düzenle',
      scrollToBottom: 'Alta kaydır',
      stop: 'Durdur',
      restorePrevious: 'Önceki kontrol noktasını geri yükle',
      restoreCheckpoint: 'Kontrol noktasını geri yükle',
      restoreFromHere: 'Kontrol noktasını geri yükle — bu istemden yeniden çalıştır',
      restoreTitle: 'Bu kontrol noktasına geri yüklensin mi?',
      restoreBody:
        'Bu istemden sonraki her şey konuşmadan kaldırılır ve istem buradan yeniden çalıştırılır.',
      restoreConfirm: 'Geri yükle ve yeniden çalıştır',
      restoreNext: 'Sonraki kontrol noktasını geri yükle',
      goForward: 'İleri git',
      sendEdited: 'Düzenlenmiş mesajı gönder',
      attachingFile: 'Ekleniyor…'
    },
    approval: {
      gatewayDisconnected: 'Hermes ağ geçidi bağlı değil',
      sendFailed: 'Onay yanıtı gönderilemedi',
      run: 'Çalıştır',
      command: 'Komut',
      moreOptions: 'Daha fazla onay seçeneği',
      allowSession: 'Bu oturuma izin ver',
      alwaysAllowMenu: 'Her zaman izin ver…',
      jumpToApproval: 'Onay gerekli',
      reject: 'Reddet',
      alwaysTitle: 'Bu komuta her zaman izin verilsin mi?',
      alwaysDescription: pattern =>
        `Bu, "${pattern}" desenini kalıcı izin listenize (~/.hermes/config.yaml) ekler. Hermes bunun gibi komutlar için bu oturumda veya gelecekteki herhangi bir oturumda tekrar sormayacaktır.`,
      alwaysAllow: 'Her zaman izin ver'
    },
    clarify: {
      notReady: 'Açıklama isteği henüz hazır değil',
      gatewayDisconnected: 'Hermes ağ geçidi bağlı değil',
      sendFailed: 'Açıklama yanıtı gönderilemedi',
      loadingQuestion: 'Soru yükleniyor…',
      other: 'Diğer (yanıtınızı yazın)',
      placeholder: 'Yanıtınızı yazın…',
      shortcutSuffix: ' ile gönderin',
      back: 'Geri',
      skip: 'Atla',
      send: 'Gönder'
    },
    tool: {
      code: 'Kod',
      copyCode: 'Kodu kopyala',
      renderingImage: 'Görsel oluşturuluyor',
      copyOutput: 'Çıktıyı kopyala',
      copyCommand: 'Komutu kopyala',
      copyContent: 'İçeriği kopyala',
      copyUrl: "URL'yi kopyala",
      copyResults: 'Sonuçları kopyala',
      copyQuery: 'Sorguyu kopyala',
      copyFile: 'Dosyayı kopyala',
      copyPath: 'Yolu kopyala',
      outputAlt: 'Araç çıktısı',
      rawResponse: 'Ham yanıt',
      copyActivity: 'Etkinliği kopyala',
      recoveredOne: '1 başarısız adımdan sonra kurtarıldı',
      recoveredMany: count => `${count} başarısız adımdan sonra kurtarıldı`,
      failedOne: '1 adım başarısız',
      failedMany: count => `${count} adım başarısız`,
      statusRunning: 'Çalışıyor',
      statusError: 'Hata',
      statusRecovered: 'Kurtarıldı',
      statusDone: 'Tamamlandı'
    }
  },

  prompts: {
    gatewayDisconnected: 'Hermes ağ geçidi bağlı değil',
    sudoSendFailed: 'Sudo parolası gönderilemedi',
    secretSendFailed: 'Gizli bilgi gönderilemedi',
    sudoTitle: 'Yönetici parolası',
    sudoDesc:
      "Hermes ayrıcalıklı bir komut çalıştırmak için sudo parolanıza ihtiyaç duyuyor. Yalnızca yerel ajanınıza gönderilir.",
    sudoPlaceholder: 'sudo parolası',
    secretTitle: 'Gizli bilgi gerekli',
    secretDesc: 'Hermes devam etmek için bir kimlik bilgisine ihtiyaç duyuyor.',
    secretPlaceholder: 'gizli değer'
  },

  desktop: {
    audioReadFailed: 'Kaydedilen ses okunamadı',
    sessionUnavailable: 'Oturum kullanılamıyor',
    createSessionFailed: 'Yeni oturum oluşturulamadı',
    promptFailed: 'İstem başarısız',
    providerCredentialRequired:
      'İlk mesajınızı göndermeden önce bir sağlayıcı kimlik bilgisi ekleyin.',
    emptySlashCommand: 'boş eğik çizgi komutu',
    desktopCommands: 'Masaüstü komutları',
    skillCommandsAvailable: count => `${count} beceri komutu kullanılabilir.`,
    warningLine: message => `uyarı: ${message}`,
    yoloArmed: 'Bu sohbet için YOLO etkinleştirildi',
    yoloOff: 'YOLO kapalı',
    yoloSystem: active => `Bu oturum için YOLO ${active ? 'açık' : 'kapalı'}`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'YOLO geçişi yapılamadı',
    profileStatus: current =>
      `Profil: ${current}. Başka bir profilde sohbet başlatmak için /profile <ad> veya "Yeni oturum" seçicisini kullanın.`,
    unknownProfile: 'Bilinmeyen profil',
    noProfileNamed: (target, available) => `"${target}" adında profil yok. Mevcut: ${available}`,
    newChatsProfile: name => `Yeni sohbetler ${name} profilini kullanacak.`,
    setProfileFailed: 'Profil ayarlanamadı',
    sttDisabled: 'Konuşmadan metne ayarlarda devre dışı.',
    stopFailed: 'Durdurma başarısız',
    regenerateFailed: 'Yeniden oluşturma başarısız',
    editFailed: 'Düzenleme başarısız',
    resumeFailed: 'Sürdürme başarısız',
    nothingToBranch: 'Dallanacak bir şey yok',
    branchNeedsChat: 'Dallanmadan önce bir sohbet başlatın veya sürdürün.',
    sessionBusy: 'Oturum meşgul',
    branchStopCurrent: 'Bu sohbeti dallandırmadan önce mevcut turu durdurun.',
    branchNoText: 'Bu mesajda dallanacak metin yok.',
    branchTitle: 'Dal',
    branchFailed: 'Dallanma başarısız',
    deleteFailed: 'Silme başarısız',
    archived: 'Arşivlendi',
    archiveFailed: 'Arşivleme başarısız',
    cwdChangeFailed: 'Çalışma dizini değişikliği başarısız',
    cwdStagedTitle: 'Çalışma dizini hazırlandı',
    cwdStagedMessage:
      'Bu etkin oturuma cwd değişikliklerini uygulamak için masaüstü arka ucunu yeniden başlatın.',
    modelSwitchFailed: 'Model değişikliği başarısız',
    sessionExported: 'Oturum dışa aktarıldı',
    sessionExportFailed: 'Oturum dışa aktarılamadı',
    imageSaved: 'Görsel kaydedildi',
    downloadStarted: 'İndirme başladı',
    restartToUseSaveImage: 'Görsel kaydetmek için Hermes Desktop\'ı yeniden başlatın.',
    restartToSaveImages: 'Görsel kaydetmek için Hermes Desktop\'ı yeniden başlatın',
    imageDownloadFailed: 'Görsel indirme başarısız',
    openImage: 'Görseli aç',
    downloadImage: 'Görseli indir',
    savingImage: 'Görsel kaydediliyor',
    imagePreviewFailed: 'Görsel önizlemesi başarısız',
    imageAttach: 'Görsel ekle',
    imageWriteFailed: 'Görsel diske yazılamadı.',
    imageAttachFailed: 'Görsel ekleme başarısız',
    attachImages: 'Görsel ekle',
    clipboard: 'Pano',
    noClipboardImage: 'Panoda görsel bulunamadı',
    clipboardPasteFailed: 'Panodan yapıştırma başarısız',
    dropFiles: 'Dosya bırak',
    handoff: {
      pickPlatform: 'Bir hedef seçin',
      success: platform => `${platform} platformuna devredildi. İstediğiniz zaman buradan devam edin.`,
      systemNote: platform =>
        `↻ ${platform} platformuna devredildi — istediğiniz zaman buradan devam edin.`,
      failed: error => `Devir başarısız: ${error}`,
      timedOut: 'Ağ geçidi beklenirken zaman aşımı. `hermes gateway` çalışıyor mu?'
    }
  },

  errors: {
    genericFailure: 'Bir şeyler ters gitti',
    boundaryTitle: 'Arayüzde bir şeyler bozuldu',
    boundaryDesc: 'Görünüm beklenmedik bir hatayla karşılaştı. Sohbetleriniz ve ayarlarınız güvende.',
    reloadWindow: 'Pencereyi yeniden yükle',
    openLogs: 'Günlükleri aç'
  },

  ui: {
    search: {
      clear: 'Aramayı temizle'
    },
    pagination: {
      label: 'sayfalama',
      previous: 'Önceki',
      previousAria: 'Önceki sayfaya git',
      next: 'Sonraki',
      nextAria: 'Sonraki sayfaya git'
    },
    sidebar: {
      title: 'Kenar çubuğu',
      description: 'Mobil kenar çubuğunu gösterir.',
      toggle: 'Kenar çubuğunu aç/kapat'
    }
  }
})
