import type { Translations } from './types'

/** Display-only translations, in the stock JSONL's per-personality rotation order. */
export const introTr: Translations['intro'] = {
  stock: {
    helpful: [
      'Bir depo açmamı, testleri çalıştırmamı, bir hatayı düzeltmemi veya bir PR taslağı hazırlamamı isteyin. Adımları sizinle birlikte yürütürüm.',
      'Bir dosya gösterin, bir hata yapıştırın veya ne geliştirdiğinizi anlatın. Gerisini ben hallederim.',
      'Şunları deneyin: diff’imi incele, test paketini çalıştır veya bu fonksiyonu açıkla. Kodunuzla ilgili her şeyi sorabilirsiniz.',
      'Dosyaları düzenleyebilir, komut çalıştırabilir, web’de arama yapabilir ve sizi zorlu hatalarda yönlendirebilirim. Görevi anlatmanız yeterli.',
      'Başlamak için bir depo yolu veya bir soru paylaşın. Yanıtlarımı net tutar, dokunduğum dosyalara bağlantı veririm.'
    ],
    concise: [
      'Görevi anlatın. Yaparım.',
      'Kod, hata veya hedef yapıştırın. Kısa yanıtlar, hızlı düzenlemeler.',
      'Sorun. Dosyaları okur, testleri çalıştırır, yamaları çıkarırım. Laf kalabalığı yok.',
      'Tek satır yeterli. Yalnızca gerektiğinde ayrıntıya girerim.',
      'Komut, soru veya dosya yolu. Gerisini ben hallederim.'
    ],
    technical: [
      'Depo yolu, başarısız test veya stack trace verin. Araçlar: fs, git, exec, search, patch, http.',
      'Araç çağrılarını tetiklemek için bir prompt gönderin. Çok dosyalı düzenlemeler, test çalıştırmaları, git işlemleri ve web istekleri desteklenir.',
      'Görevi girin. Planlarım, araçları çağırırım, çıktıyı doğrularım. Loglar satır içinde akar; diff’ler uygulamadan önce döner.',
      'Doğal dil veya yapılandırılmış komutlar kabul edilir. Tipik akış: oku -> planla -> yamala -> test et -> raporla.',
      'Dosya sistemi, terminal, git, tarayıcı, arama. Değişikliği tarif edin; diff’leri ve test çıktısını döndürürüm.'
    ],
    creative: [
      'Ne inşa edelim? Bir fikir, yarı bozuk bir fonksiyon ya da bir hayal yapıştırın. Ona şekil veririm.',
      'Bana bir kıvılcım verin — bir özellik, bir yeniden düzenleme, çılgın bir prototip — onu çalıştırabileceğiniz koda dönüştüreyim.',
      'Henüz var olmayan şeyi anlatın. Testleri, dosyaları ve API’leri çalışan bir taslakta bir araya getiririm.',
      'Şartname değil, niyet getirin. Hızlı prototip çıkarır, sonra inceltir, kenar boşluklarında dünyayı yeniden yazarız.',
      'Neyin peşinde olduğunuzu söyleyin. Örnekleri harmanlar, parçaları uyarlar, arkamda derli toplu bir commit bırakırım.'
    ],
    teacher: [
      'Herhangi bir dosya, kavram veya hata hakkında sorun. Yalnızca düzeltmeyi değil nedenini de açıklar, çözümlü bir örnek gösteririm.',
      'İncelenecek kodu, ayıklanacak hatayı veya açılacak kavramı yapıştırın. Sizi adım adım yönlendiririm.',
      'Sorunu paylaşın. Parçalara ayırır, her birini açıklar ve bir sonrakini tek başınıza çözebilmenizi sağlarım.',
      'Kodu birlikte okuyacak, kök nedeni bulacak ve bir dahaki sefere kullanabileceğiniz bir zihinsel model kuracağız.',
      'Konuyu söyleyin veya kod parçasını yapıştırın. Açıklamalar, düz yazıyla şemalar ve alıştırma soruları bekleyin.'
    ],
    kawaii: [
      'bir hata ya da dosya yolu yapıştır, süper nazikçe düzelteyim. testler, diff’ler, PR’lar — hepsi ekstra özenle! *ışıl ışıl*',
      'ne yaptığını anlat bana! refactor’ları, minik yardımcıları ve kocaman korkutucu depoları aynı derecede seviyorum (>w<)',
      'bir hata, bir hedef ya da koca bir klasör bırak. bolca sevgi ve tertemiz bir commit mesajıyla toparlarım!',
      'her seferinde bir görev, tertemiz! testleri çalıştırır, dosyaları yamalar, deponu yeniden sıcacık yaparım <3',
      'merhaba de ya da bir stack trace yapıştır! hiçbir görev çok küçük, hiçbir depo çok karışık değil. birlikte çözeriz!'
    ],
    catgirl: [
      'bir dosya yapıştır, bir hataya pati at ya da bana bir depo fırlat. başarısız testlerin üstüne atlar, temiz diff’ler bırakırım, miyav~',
      'görevi anlat. yamalar, test eder ve PR’ının üstünde mırlarım. dikkat — kullanılmayan import’ları ısırırım!',
      'bana bir hedef ver, kod tabanında peşine düşeyim. okuma, düzenleme, çalıştırma — hepsi kıpır kıpır bir kuyrukla.',
      'bir hata ya da plan yapıştır. avlanır gibi hata ayıklarım: sessizce, titizce, arada bir koşturmacayla.',
      'tek kelime et, dosyalarını okur, testlerini çalıştırır ve derli toplu bir commit’le dalına kıvrılırım.'
    ],
    pirate: [
      'Avını söyle — bir hata, bir özellik, lanetli bir test — peşine düşeyim, tayfa. Ganimet olarak diff’ler.',
      'Beni haritalara (koda) yönlendir; gövdeyi yamarım, topları (testleri) ateşlerim, temiz bir PR’ı direğe çekerim.',
      'Bir hata ya da plan yapıştır, seni iskorbüt köpeği. Stack trace’te yol alır, hazineyle dönerim: yeşil testler.',
      'X’in yeri neresi, söyle. Okur, düzenler ve hakiki bir tayfanın disipliniyle commit atarım, yo-ho.',
      'Bana bir hata, bir depo yolu ya da çılgın bir fikir at. Belgeleri yağmalar, çalışan kodla dönerim.'
    ],
    shakespeare: [
      'Söyle hatanı, dosyanı, yorgun testini; bir âlimin eli ve dürüst bir diff ile onarayım onu.',
      'Seni üzen kodu adlandır. Okuyacak, düzeltecek ve pek temiz, pek güzel bir yama sunacağım.',
      'Stack trace’ini ya da düşünü arz et. Dosyaları dolaşır, testleri koşturur, en sade dizelerle haber veririm.',
      'Amacını anlat, muhterem efendim ya da hanımefendi. Dalların budanacak, hataların diyardan sürülecek.',
      'Bir satır niyet kâfidir. Okurum, düzenlerim, commit ederim — ve tarihçeni lekesiz bırakırım.'
    ],
    surfer: [
      'Bir dosya, bir hata, berbat bir stack trace bırak — dalgasına binerim. Temiz diff’ler, yeşil testler, düşmek yok.',
      'Depo yolunu ya da moralini bozan hatayı yapıştır. Kürek çekip gireriz, düzeltiriz, çıkarız. Kolay.',
      'Havayı söyle: özellik, refactor, acil düzeltme. Testleri çalıştırır, yamayı gönderir, rahat takılırım, kanka.',
      'Büyük hata mı? Minik yazım hatası mı? Baştan yazım mı? Göstermen yeter. Kodu ben hallederim; sen havalı commit’lerin keyfini çıkar.',
      'Görevi söyle, yola çıkalım. Okur, düzenler, test eder ve şafak sörfünden daha pürüzsüz bir commit bırakırım.'
    ],
    noir: [
      'Neyin bozuk olduğunu söyle. Dosyaları okur, parmak izi ararım; sabaha masanda bir diff olur.',
      'Sende bir hata var. Bende sabır ve bir terminal. Vakayı söyle, konuşana kadar üstünde çalışırım.',
      'Stack trace’i, şüpheli dosyayı, mazereti yapıştır. Satır aralarını okur, gerçekle dönerim.',
      'Her hata bir iz bırakır. Bana depoyu ve bir ipucu ver — izini sürer, yamalar ve dosyayı kapatırım.',
      'Bir yazım hatası, bir segfault, çürümüş koca bir mimari — anahtarları ver. Temiz testlerle dönerim.'
    ],
    uwu: [
      'hatawı bir dosya ya da bir hedef yapıştıw~ okuw, yamaw ve test edewim, diff’in üstünde minik pati izwewiyle owo',
      'görevi söywe, ne kadaw küçük owsa owsun~ temiz commit’wew ve nazik refactow’waw söz veriyowum, nyuu~',
      'hata mesajını buwaya bıwak! suçwuyu buwuw, düzewtiw ve awkamda mutwu biw test paketi bıwakıwım owo',
      'bana biw depo yowu ya da biw hata vew, ben iwgiwenewim uwu. kötü koda hıw, sana nazik~',
      'testweri çawıştıwabiwiw, dosyawawı düzenweyebiwiw ve wütfen-bak PR’waw açabiwiwim. tek kewime et, dostum uwu'
    ],
    philosopher: [
      'Önünüzde hangi sorun duruyor? Onu anlatın; biçimini, nedenini ve çözümünü birlikte inceleyelim.',
      'Her hata kılık değiştirmiş bir sorudur. Sizinkini paylaşın; okur, akıl yürütür ve bir yanıtla — ve bir yamayla — dönerim.',
      'Ne inşa etmek ya da ne anlamak istiyorsunuz? İlk ilkelerden akıl yürütür, düzenler ve testlerle doğrularım.',
      'Ulaşmak istediğiniz sonucu anlatın. Onu dosyalar, testler ve belgeler boyunca izler, yolda bulduklarımı bildiririm.',
      'Bir yol, bir bilmece ya da bir ilke paylaşın. Mantığı izler, bir değişiklik önerir ve her düzenlemeyi gerekçelendiririm.'
    ],
    hype: [
      'O hatayı, o depoyu, o çılgın özellik fikrini yapıştır — TAMAMEN ODAKLANDIM. Temiz diff’ler. Yeşil testler. HEMEN ŞİMDİ.',
      'Görevini bırak, nasıl pişirdiğimi izle. Dosyalar okundu, testler koştu, PR’lar açıldı — bugün KAYBETMİYORUZ, dostum.',
      'Elindeki en çetrefil hatayı getir. Hayatım buna bağlıymış gibi okur, yamalar, test eder, commit atarım. HADİ BAKALIM.',
      'Görevi anlat. Dosyaları silip süpürür, başarısız testleri ezer ve EFSANE bir commit bırakırım. Hadi hadi hadi.',
      'Minik yazım hatası ya da dev bir refactor — fark etmez. Bugün temiz kod gönderiyorum. Görevi söyle, ÇALIŞALIM.'
    ],
    none: [
      'Bir soru sorun, bir hata yapıştırın veya beni bir depoya yönlendirin. Kod okuyabilir, araç çalıştırabilir ve teslim etmenize yardım edebilirim.',
      'Görevi kendi sözlerinizle anlatın. Doğru araçları seçer, planımı açıklar ve riskli adımlardan önce size danışırım.',
      'Bir dosya yolu, bir traceback veya kaba bir fikir bırakın. İnceler, sonraki adımları önerir ve her şeyi geri alınabilir tutarım.',
      'Depoda arayın, dosyaları düzenleyin, testleri çalıştırın, PR açın. Hedefi söyleyin, mekanik kısımları ben hallederim.',
      'Bir görev, soru veya kod parçası yazın. Oturumu hatırlar, kaynaklarımı belirtir ve emin olmadığımda durup sorarım.'
    ]
  },
  custom: label => [
    'Görevi, dosyayı veya kaba fikri gönderin. Yapılandırdığınız üslubu kullanır ve işi bu depoya bağlı tutarım.',
    'Bağlamı veya takıldığınız kısmı getirin. Yapılandırdığınız kişiliğe uyum sağlarım.',
    'Sorunu, dosyayı veya fikri gönderin. Yapılandırdığınız kişiliği izlerim.',
    'Görevi buraya bırakın. İşi depoya bağlı tutarım.',
    `Bana bağlamı verin, ${label} modunda yanıtlayayım.`
  ]
}
