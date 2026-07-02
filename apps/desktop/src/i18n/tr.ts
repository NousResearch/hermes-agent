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
  }
})

export { tr }
