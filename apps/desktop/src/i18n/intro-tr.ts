import type { Translations } from './types'

/** Display-only translations, in the stock JSONL's per-personality rotation order. */
export const introTr: Translations['intro'] = {
  stock: {
    helpful: [
      'Bir repo açmamı, test çalıştırmamı, bir hatayı düzeltmemi ya da PR taslağı hazırlamamı iste. Adımları birlikte geçeriz.',
      'Bana bir dosya göster, bir hata yapıştır ya da ne yaptığını anlat. Gerisini bana bırak.',
      'Dene: diff’imi incele, test paketini çalıştır ya da bu işlevi açıkla. Kodunla ilgili her şeyi sor.',
      'Dosya düzenleyebilir, komut çalıştırabilir, web’de arayabilir ve zor hatalarda sana yol gösterebilirim. Görevi anlatman yeterli.',
      'Başlamak için bir repo yolu ya da soru paylaş. Yanıtlarım açık olur, değiştirdiğim dosyalara bağlantı veririm.'
    ],
    concise: [
      'Görevi anlat. Gerisini hallederim.',
      'Kod, hata ya da hedef yapıştır. Kısa yanıtlar, hızlı düzenlemeler.',
      'Sor. Dosyaları okur, test çalıştırır, yama gönderirim. Laf kalabalığı yok.',
      'Tek satır yeter. Ancak önemliyse uzatırım.',
      'Komut, soru ya da dosya yolu. Gerisi bende.'
    ],
    technical: [
      'Repo yolu, başarısız test ya da stack trace ver. Araçlar: fs, git, exec, search, patch, http.',
      'Araç çağrılarını tetiklemek için komut gönder. Çok dosyalı düzenlemeler, test çalışmaları, git işlemleri ve web çekmeleri desteklenir.',
      'Görevi gir. Planlar, araç çağırır, çıktıyı doğrularım. Günlükler satır içinde akar; diff’ler uygulamadan önce gelir.',
      'Doğal dil ya da yapılandırılmış komut kabul edilir. Tipik akış: oku -> planla -> yama -> test et -> raporla.',
      'Dosya sistemi, terminal, git, tarayıcı, arama. Değişikliği anlat; diff ve test çıktılarıyla dönerim.'
    ],
    creative: [
      'Ne yapıyoruz? Bir fikir, yarım bozuk bir işlev ya da bir hayal yapıştır. Şekle sokarım.',
      'Bana bir kıvılcım ver — bir özellik, bir yeniden düzenleme, çılgın bir prototip — çalışır koda dönüştürürüm.',
      'Henüz var olmayan şeyi anlat. Testleri, dosyaları ve API’leri çekip çalışan bir taslağa dönüştürürüm.',
      'Spesifikasyon değil, niyet getir. Hızlı prototipler, sonra inceleriz; dünyayı kenar boşluklarında yeniden yazarız.',
      'Neyin peşinde olduğunu söyle. Örnekleri harmanlar, parçaları uyarlar, arkada temiz bir commit bırakırım.'
    ],
    teacher: [
      'Herhangi bir dosya, kavram ya da hata hakkında sor. Sadece çözümü değil, nedenini açıklar, çözülmüş bir örnek gösteririm.',
      'İncelenecek kod, ayıklanacak hata ya da açılacak kavram yapıştır. Adım adım yol gösteririm.',
      'Sorunu paylaş. Parçalara böler, her birini açıklar, bir sonrakini tek başına çözebilmeni sağlarım.',
      'Kodu birlikte okur, kök nedeni bulur, tekrar kullanabileceğin bir zihinsel model kurarız.',
      'Konuyu söyle ya da parçayı yapıştır. Açıklamalar, sözlü diyagramlar ve alıştırma soruları seni bekler.'
    ],
    kawaii: [
      'bir hata ya da dosya yolu yapıştır, nazikçe düzeltirim. testler, diff’ler, PR’lar — hepsi ekstra özenle! *parıltı*',
      'söyle ne yapıyorsun! yeniden düzenlemeleri, minik yardımcıları ve kocaman korkunç repoları da severim (>w<)',
      'bir hata, bir hedef ya da koca bir klasör bırak. sevgiyle toparlar, temiz bir commit mesajı yazarım!',
      'tek seferde bir görev, tertemiz! test çalıştırır, dosya yamalar, reponu sıcacık yaparım <3',
      'merhaba de ya da stack trace yapıştır! küçük görev yok, karışık repo yok. birlikte çözeriz!'
    ],
    catgirl: [
      'bir dosya yapıştır, hataya pati at ya da bana repo fırlat. başarısız testlerin üstüne atlar, temiz diff’ler bırakırım, nyan~',
      'görevi anlat. yamalar, test eder, PR’ının üstünde mırlarım. dikkat — kullanılmayan import’ları kemiririm!',
      'bana hedef ver, kod tabanında kovalarım. okur, düzenler, çalıştırırım — hepsi seğiren kuyrukla.',
      'hata ya da plan yapıştır. avlanır gibi ayıklarım: sessiz, titiz, arada zıp zıp.',
      'tek kelime yeter; dosyalarını okur, testlerini çalıştırır, düzenli bir commit’le dalına kıvrılırım.'
    ],
    pirate: [
      'Avını söyle — hata, özellik, lanetli test — peşine düşerim, tayfa. Ganimet olarak diff’ler.',
      'Haritaları (kodu) göster, gövdeyi yamalar, topları (testleri) ateşler, temiz bir PR çekerim.',
      'Hata ya da plan yapıştır, seni kara faresi. stack trace’de yol alır, hazineyi getiririm: yeşil testler.',
      'X’in nerede olduğunu söyle. Gerçek mürettebat disipliniyle okur, düzenler, commit’lerim, arrr.',
      'Bana hata, repo yolu ya da çılgın fikir fırlat. Dökümanları yağmalar, çalışan kodla dönerim.'
    ],
    shakespeare: [
      'Söyle hatanı, dosyanı, yorgun testini; âlim eliyle, dürüst diff ile derman olurum.',
      'Seni üzen kodu söyle. Okur, düzeltir, tertemiz bir yama sunarım.',
      'Stack trace’ini ya da hayalini ser önüme. Dosyaları gezer, test çalıştırır, sade mısralarla rapor ederim.',
      'Maksadını anlat, ey asilzade. Dalların budanacak, hataların diyarından kovulacak.',
      'Tek satır niyet yeter. Okurum, değiştiririm, commit’lerim — tarihin lekesiz kalır.'
    ],
    surfer: [
      'Dosya, hata, fena stack trace fırlat — üstünde sörf yaparım. Temiz diff’ler, yeşil testler, silinme yok.',
      'Repo yolunu ya da moral bozan hatayı yapıştır. Kürek çeker girer, düzeltir, çıkarız. Kolay iş.',
      'Havayı söyle: özellik, yeniden düzenleme, hotfix. Test çalıştırır, yamayı gönderir, rahat takılırım kanka.',
      'Büyük hata? Minik yazım yanlışı? Komple yeniden yazım? Göster yeter. Kodu ben hallederim; sen commit’lerle keyfine bak.',
      'Görevi söyle, başladık. Okur, düzenler, test eder, şafak devriyesinden pürüzsüz commit bırakırım.'
    ],
    noir: [
      'Neyin bozuk olduğunu söyle. Dosyaları okur, parmak izi toplar, sabaha masana diff bırakırım.',
      'Sende hata var. Bende sabır ve terminal. Davayı söyle, konuşana kadar üstünde çalışırım.',
      'Stack trace’i, şüpheli dosyayı, mazereti yapıştır. Satır aralarını okur, gerçekle dönerim.',
      'Her hata iz bırakır. Repoyu ve ipucunu ver — izi sürer, yamalar, dosyayı kapatırım.',
      'Yazım yanlışı, segfault, çürümüş mimari — anahtarları ver. Temiz testlerle dönerim.'
    ],
    uwu: [
      'hatawı dosya ya da hedef yapıştıw~ okuwum, yamawawım, test edewim; diff’te minik pati izwewiyle owo',
      'göwevi söywe, ne kadaw küçük owursa owsun~ temiz commit’wew ve nazik yeniden düzenlemewew söz, nyuu~',
      'hata mesajını buwada bıwak! suçwuyu buwuw, düzeltiv ve awkasında mutwu biw test paketi biwakırım owo',
      'bana wepo youu ya da hata vev, iwgiwenirim uwu. kötü koda gwrrr, sana sevgi~',
      'test çawıştıwabiwir, dosya düzenweyebiwir, winca incewe PR’waw açabiwirim. tek kewime söywe, dostum uwu'
    ],
    philosopher: [
      'Önünde hangi sorun duruyor? Anlat, biçimini, nedenini ve çözümünü birlikte inceleyelim.',
      'Her hata kılık değiştirmiş sorudur. Seninkini paylaş; okur, düşünür, yanıtla — ve yamayla dönerim.',
      'Ne yapmak ya da anlamak istersin? İlk ilkelerden akıl yürütür, düzenler, testle doğrularım.',
      'Erişmek istediğin sonu anlat. Dosyalarda, testlerde, dökümanda izini sürer, yolda bulduklarımı rapor ederim.',
      'Bir yol, bilmece ya da ilke paylaş. Mantığı izler, değişiklik önerir, her düzenlemeyi gerekçelendiririm.'
    ],
    hype: [
      'VER HATAYI, REPOYU, ÇILGIN FİKRİ — TAM GAZ GELİYORUM. Temiz diff’ler. Yeşil testler. HEMEN.',
      'Görevi bırak, izle. Dosyalar okundu, testler çalıştı, PR’ler açıldı — bugün KAYBETMİYORUZ dostum.',
      'En gıcık hatanı getir. Canım pahasına okur, yamalar, test eder, commit’lerim. HAYDİ.',
      'Görevi anlat. Dosyaları süpürür, başarısız testleri ezer, ORTALIĞI YIKAN commit bırakırım. Haydi haydi haydi.',
      'Minik yazım yanlışı ya da dev yeniden düzenleme — fark etmez. Bugün temiz kod gönderiyorum. Görevi söyle, ÇALIŞALIM.'
    ],
    none: [
      'Soru sor, hata yapıştır ya da repo göster. Kod okur, araç çalıştırır, göndermene yardım ederim.',
      'Görevi kendi cümlelerinle anlat. Doğru araçları seçer, planımı açıklar, riskli adımlarda sorarım.',
      'Dosya yolu, traceback ya da kabataslak fikir bırak. İnceler, sonraki adımları önerir, her şeyi geri alınabilir tutarım.',
      'Repoda ara, dosya düzenle, test çalıştır, PR aç. Hedefi söyle, mekanik kısmı ben hallederim.',
      'Görev, soru ya da parça yaz. Oturumu hatırlar, kaynaklarımı belirtir, emin olmadığımda durup sorarım.'
    ]
  },
  custom: label => [
    'Görevi, dosyayı ya da kabataslak fikri gönder. Yapılandırdığın sesi kullanır, işi bu repoya yakın tutarım.',
    'Bağlamı ya da takıldığın yeri getir. Yapılandırdığın kişiliğe uyum sağlarım.',
    'Sorunu, dosyayı ya da fikri gönder. Yapılandırdığın kişiliği takip ederim.',
    'Görevi buraya bırak. İşi repoya yakın tutarım.',
    `Bana bağlamı ver, ${label} kipinde yanıtlayayım.`
  ]
}
