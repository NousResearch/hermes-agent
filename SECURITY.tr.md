# Hermes Agent Güvenlik Politikası

Bu belge, Hermes Agent'in güven modelini tanımlar, projenin yük taşıyıcı
olarak kabul ettiği tek güvenlik sınırını belirtir ve güvenlik açığı
bildirimlerinin kapsamını tanımlar.

## 1. Güvenlik Açığı Bildirme

[GitHub Güvenlik Danışma Sayfası](https://github.com/NousResearch/hermes-agent/security/advisories/new)
üzerinden veya **security@nousresearch.com** adresine özel olarak bildirin.
Güvenlik açıkları için herkese açık sorunlar (issue) açmayın. **Hermes
Agent bir hata ödülü programı (bug bounty) işletmemektedir.**

Yararlı bir bildirim şunları içerir:

- Kısa bir açıklama ve ciddiyet değerlendirmesi.
- Dosya yolu ve satır aralığı ile tanımlanmış etkilenen bileşen
  (örn. `path/to/file.py:120-145`).
- Ortam bilgileri (`hermes version`, commit SHA, işletim sistemi, Python
  sürümü).
- `main` veya en son sürüm üzerinde bir yeniden üretim adımı.
- Bölüm 2'deki hangi güven sınırının (trust boundary) aşıldığının
  belirtilmesi.

Lütfen göndermeden önce Bölüm 2 ve Bölüm 3'ü okuyun. Bu politikanın
sınır olarak ele almadığı bir süreç içi sezgisel yöntemin (in-process
heuristic) sınırlarını gösteren raporlar, Bölüm 3 kapsamında
kapsam dışı olarak kapatılacaktır — ancak Bölüm 3.2'ye bakın: bu tür
raporlar normal sorunlar (issue) veya çekme talepleri (pull request)
olarak hâlâ memnuniyetle karşılanır; sadece özel güvenlik kanalı
üzerinden değil.

---

## 2. Güven Modeli

Hermes Agent, tek kiracılı (single-tenant) kişisel bir ajandır. Duruşu
katmanlıdır ve katmanlar eşit derecede yük taşıyıcı değildir. Bildirenler
ve işletmeciler, bunları aynı terimlerle değerlendirmelidir.

### 2.1 Tanımlar

- **Ajan süreci (Agent process).** Hermes Agent'i çalıştıran Python
  yorumlayıcısı, yüklediği tüm Python modülleri dahil (yetenekler/skills,
  eklentiler/plugins, hook işleyicileri/hook handlers).
- **Terminal arka ucu (Terminal backend).** `terminal()` aracı için
  takılabilir bir yürütme hedefi. Varsayılan, komutları doğrudan ana
  bilgisayarda (host) çalıştırır. Diğer arka uçlar komutları bir konteyner,
  bulut kum havuzu (cloud sandbox) veya uzak ana bilgisayar içinde
  çalıştırır.
- **Girdi yüzeyi (Input surface).** Ajanın bağlamına içeriğin girdiği
  herhangi bir kanal: işletmeci girdisi, web getirmeleri (web fetches),
  e-posta, ağ geçidi (gateway) mesajları, dosya okumaları, MCP sunucu
  yanıtları, araç sonuçları.
- **Güven zarfı (Trust envelope).** Bir işletmecinin Hermes Agent'i
  çalıştırarak ona örtülü olarak erişim izni verdiği kaynaklar kümesi —
  genellikle, işletmecinin kendi kullanıcı hesabının ana bilgisayarda
  erişebildiği her şey.
- **Duruş (Stance).** Hermes Agent'in belgelerinde veya kodunda, tüketici
  bir katmanın (adaptör/adapter, kullanıcı arayüzü/UI, dosya yazıcı/file
  writer, kabuk/shell) ajan çıktısını nasıl ele alması gerektiğine dair
  açık bir ifade — örn. "gösterge panosu (dashboard), ajan çıktısını
  etkin olmayan HTML olarak işler."

### 2.2 Sınır: İşletim Sistemi Düzeyinde İzolasyon

**Düşmanca/saldırgan bir YDM'ye (LLM) karşı tek güvenlik sınırı işletim
sistemidir.** Ajan sürecinin içindeki hiçbir şey sınırlama (containment)
oluşturmaz — onay kapısı (approval gate), çıktı düzeltme (output
redaction), herhangi bir desen tarayıcı (pattern scanner), herhangi bir
araç izin listesi (tool allowlist) dahil. YDM çıktısını tarayan herhangi
bir süreç içi bileşen, bir saldırgan tarafından etkilenmiş bir dize
(string) üzerinde çalışan sezgisel bir yöntemdir (heuristic) ve bu
politika onu bu şekilde ele alır.

Hermes Agent iki işletim sistemi düzeyinde izolasyon duruşunu
destekler. Bunlar farklı tehditleri ele alır ve bir işletmeci bilinçli
bir seçim yapmalıdır.

#### Terminal-arka ucu izolasyonu

Varsayılan olmayan bir terminal arka ucu, YDM tarafından üretilen kabuk
komutlarını bir konteyner, uzak ana bilgisayar veya bulut kum havuzu
içinde çalıştırır. Dosya araçları (`read_file`, `write_file`, `patch`)
da bu arka uç üzerinden çalışır, çünkü bunlar kabuk sözleşmesi (shell
contract) üzerinde uygulanmıştır — arka ucunun sunmadığı yollara
erişemezler.

Bunun sınırladığı: ajanın kabuk veya dosya işlemleri yayınlayarak
yaptığı her şey. Bunun **sınırlamadığı**: ajanın kendi Python sürecinde
yaptığı her şey. Bu, kod-yürütme aracını (ana bilgisayarda bir alt süreç
olarak başlatılan), MCP alt süreçlerini (ajanın ortamından başlatılan),
eklenti yüklemeyi, hook dağıtımını ve yetenek yüklemeyi (hepsi ajan
yorumlayıcısına aktarılır) içerir.

Terminal-arka ucu izolasyonu, endişenin YDM tarafından üretilen yıkıcı
kabuk komutları veya istenmeyen dosya aracı yazmaları olduğu ve
işletmecinin diğer durumlarda güvenilir olduğu durumlarda doğru
duruştur.

#### Tüm-süreç sarmalama

Tüm-süreç sarma, ajan süreç ağacının tamamını bir kum havuzu (sandbox)
içinde çalıştırır. Her kod yolu — kabuk, kod-yürütme, MCP, dosya
araçları, eklentiler, hooklar, yetenek yükleme — aynı dosya sistemi,
ağ, süreç ve (uygun olduğunda) çıkarım (inference) politikasına
tabidir.

Hermes Agent bunu iki şekilde destekler:

- **Hermes Agent'in kendi Docker imajı ve Compose kurulumu.** Daha
  hafiftir; ajan, işletmeci tarafından yapılandırılmış bağlama
  noktaları (mounts) ve ağ politikası ile standart bir konteynerde
  çalışır.
- **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell).**
  OpenShell, dosya sistemi, ağ (L7 giden/egress), süreç/sistem
  çağrısı ve çıkarım yönlendirme katmanları arasında bildirimsel
  (declarative) politika ile oturum başına kum havuzları sağlar. Ağ
  ve çıkarım politikaları sıcak yeniden yüklenebilir (hot-reloadable).
  Kimlik bilgileri bir Sağlayıcı (Provider) deposundan enjekte edilir
  ve kum havuzu dosya sistemine asla dokunmaz.

Tüm-süreç sarmalayıcı altında, Hermes Agent'in süreç içi sezgisel
yöntemleri (Bölüm 2.4), gerçek bir sınırın üzerine katmanlanmış kaza
önleme işlevi görür. Bu, ajanın işletmecinin kontrol etmediği
yüzeylerden — açık web, gelen e-posta, çok kullanıcılı kanallar,
güvenilmeyen MCP sunucuları — içerik aldığı ve üretim veya paylaşımlı
dağıtımlar için desteklenen duruştur.

Varsayılan yerel arka ucu güvenilmeyen girdi yüzeyleri ile çalıştıran
veya bir terminal-arka ucu kum havuzu çalıştırıp bunun kabuktan
geçmeyen kod yollarını sınırlamasını bekleyen işletmeciler, desteklenen
güvenlik duruşunun dışında işlem yapmaktadır.

### 2.3 Kimlik Bilgisi Kapsamı

Hermes Agent, daha düşük güvenli süreç içi bileşenlerine geçirdiği
ortamı (environment) filtreler: kabuk alt süreçleri, MCP alt süreçleri,
zamanlanmış görev betikleri (cron job scripts) ve kod-yürütme alt
süreci. Sağlayıcı API anahtarları ve ağ geçidi (gateway) belirteçleri
(token) gibi kimlik bilgileri varsayılan olarak temizlenir; işletmeci
veya yüklenmiş bir yetenek (skill) tarafından açıkça bildirilen
değişkenler geçirilir.

Bu, sıradan sızdırmayı (casual exfiltration) azaltır. Bu bir sınırlama
değildir. Ajan sürecinin içinde çalışan herhangi bir bileşen
(yetenekler, eklentiler, hook işleyicileri), ajanın okuyabildiği her
şeyi, bellekteki kimlik bilgileri dahil, okuyabilir. Güvenliği ihlal
edilmiş bir süreç içi bileşene karşı önlem, ortam temizleme (environment
scrubbing) değil, kurulumdan önce işletmeci incelemesidir (Bölüm 2.4,
Bölüm 2.5).

### 2.4 Süreç İçi Sezgisel Yöntemler

Aşağıdaki bileşenler YDM davranışını tarar veya uyarır. Bunlar
kullanışlıdır. Bunlar sınır değildir.

- **Onay kapısı (Approval gate)** yaygın yıkıcı kabul desenlerini tespit
  eder ve yürütmeden önce işletmeciyi uyarır. Kabuk Turing
  bütünlüğündedir (Turing-complete); kabuk dizeleri üzerinde bir
  yasak listesi (denylist) yapısal olarak eksiktir. Kapı, işbirlikçi
  mod hatalarını yakalar, düşmanca/saldırgan çıktıyı değil.
- **Çıktı düzeltme (Output redaction)** sır benzeri desenleri
  görüntülemeden gizler. Motive olmuş bir çıktı üreticisi onu alt
  edecektir.
- **Yetenekler Koruyucusu (Skills Guard)** yüklenebilir yetenek
  içeriğini enjeksiyon desenlerine karşı tarar. Bu bir inceleme
  yardımcısıdır; üçüncü taraf yetenekler için sınır, kurulumdan
  önce işletmeci incelemesidir. Bir yeteneği incelemek, yalnızca
  SKILL.md açıklamasını değil, Python kodunu ve betiklerini okumak
  anlamına gelir — yetenekler, içe aktarma (import) anında
  keyfi Python kodu çalıştırır.

### 2.5 Eklenti Güven Modeli

Eklentiler, ajan sürecine yüklenir ve tam ajan ayrıcalıklarıyla
çalışır: aynı kimlik bilgilerini okuyabilir, aynı araçları
çağırabilir, aynı hookları kaydedebilir ve ağaç içinde (in-tree)
gönderilen herhangi bir şeyle aynı modülleri içe aktarabilirler.
Üçüncü taraf eklentiler için sınır, kurulumdan önce işletmeci
incelemesidir — yeteneklerle aynı kural (Bölüm 2.4), ancak ayrıca
belirtilmiştir çünkü eklentiler mimari olarak daha ağırdır ve
genellikle kendi arka plan hizmetlerini, ağ dinleyicilerini ve
bağımlılıklarını beraberinde getirir.

Kötü niyetli veya hatalı bir eklenti, Hermes Agent'in kendisinde bir
güvenlik açığı değildir. Hermes Agent'in eklenti-kurulum veya
eklenti-bulma yolundaki, işletmecinin ne yüklediğini görmesini
engelleyen hatalar, Bölüm 3.1 kapsamındadır.

### 2.6 Dış Yüzeyler

**Dış yüzey (External surface)**, yerel ajan süreci dışında, bir
arayanın ajan işi gönderebildiği, onayları çözümleyebildiği veya ajan
çıktısı alabildiği herhangi bir kanaldır. Her yüzeyin kendi
yetkilendirme modeli vardır, ancak aşağıdaki kurallar tek tip olarak
uygulanır.

**Hermes Agent'deki Yüzeyler:**

- **Ağ geçidi platform adaptörleri (Gateway platform adapters).**
  `gateway/platforms/` dizinindeki mesajlaşma entegrasyonları
  (Telegram, Discord, Slack, e-posta, SMS, vb.) ve eklenti olarak
  gönderilen benzer adaptörler.
- **Ağa açık HTTP yüzeyleri.** API sunucu adaptörü, gösterge paneli
  (dashboard) eklentisi, kanban (kanban) eklentisinin HTTP uç
  noktaları ve dinleme soketi bağlayan diğer eklentiler.
- **Editör / IDE adaptörleri.** ACP adaptörü (`acp_adapter/`) ve yerel
  bir istemci sürecinden istek kabul eden eşdeğer entegrasyonlar.
- **TUI ağ geçidi (`tui_gateway/`).** Ink terminal kullanıcı arayüzü
  için yerel IPC üzerinden erişilen JSON-RPC arka ucu.

**Tek tip kurallar:**

1. **Bir güven sınırını geçen her yüzeyde yetkilendirme zorunludur.**
   Mesajlaşma ve ağ HTTP yüzeyleri için sınır ağdır: yetkilendirme,
   işletmeci tarafından yapılandırılmış bir arayan izin listesi
   (caller allowlist) anlamına gelir. Editör ve yerel IPC yüzeyleri
   (ACP, TUI ağ geçidi) için sınır, ana bilgisayarın kullanıcı
   hesabıdır: yetkilendirme, işletim sistemi düzeyinde erişim
   kontrolüne (dosya izinleri, yalnızca döngübaşı/loopback-only
   bağlamaları) güvenmek ve yüzeyi açık bir ağ yetkilendirme katmanı
   olmadan yerel kullanıcının ötesine sunmamak anlamına gelir.
2. **Etkinleştirilmiş her ağa açık adaptör için bir izin listesi
   (allowlist) gereklidir.** Adaptörler, bir izin listesi
   ayarlanana kadar ajan işi göndermeyi, onayları çözümlemeyi veya
   çıktıyı iletmeyi reddetmelidir. Hiçbir izin listesi
yapılandırılmadığında erişime izin vererek başarısız olan (fail open) kod
yolları, Bölüm 3.1 kapsamındaki kod hatalarıdır.
3. **Oturum tanımlayıcıları yönlendirme tutamaçlarıdır (routing
   handles), yetkilendirme sınırları değildir.** Başka bir arayanın
   oturum kimliğini bilmek, onların onaylarına veya çıktısına erişim
   sağlamaz; yetkilendirme her zaman izin listesine (veya işletim
   sistemi düzeyindeki eşdeğerine) karşı yeniden kontrol edilir.
4. **Yetkilendirilmiş küme içinde, tüm arayanlar eşit derecede
   güvenilirdir.** Hermes Agent, tek bir adaptör içinde arayan
   başına yetenekleri modellemez. Yetenek ayrımına ihtiyaç duyan
   işletmeciler, ayrı izin listeleriyle ayrı ajan örnekleri
   çalıştırmalıdır.
5. **Yalnızca yerel bir yüzeyi döngübaşı olmayan bir arayüze
   bağlamak, bir kırılma-camı (break-glass) işletmeci kararıdır
   (Bölüm 3.2).** Gösterge paneli ve diğer eklenti HTTP sunucuları
   varsayılan olarak döngübaşınadır; bunları `--host 0.0.0.0` veya
   eşdeğeri ile sunmak, genel sunum sertleştirmesini (Bölüm 4)
   işletmecinin sorumluluğu haline getirir.

---

## 3. Kapsam

### 3.1 Kapsam Dahilinde

- Bildirilen bir işletim sistemi düzeyinde izolasyon duruşundan
  (Bölüm 2.2) kaçış: saldırgan kontrollü bir kod yolunun, duruşun
  sınırladığını iddia ettiği duruma ulaşması.
- Yetkisiz dış yüzey erişimi: yapılandırılmış yetkilendirme
  kümesinin (izin listesi veya yerel IPC yüzeyleri için işletim
  sistemi düzeyindeki eşdeğeri) dışındaki bir arayanın iş
  göndermesi, çıktı alması veya onayları çözümlemesi (Bölüm 2.6).
- Kimlik bilgisi sızdırma (Credential exfiltration): işletmeci kimlik
  bilgilerinin veya oturum yetkilendirme materyalinin, bunu
  engellemesi gereken bir mekanizma (ortam temizleme hatası, adaptör
  günlüğü, kimlik bilgilerini bir üst kaynağa gönderen taşıma hatası,
  vb.) aracılığıyla güven zarfının dışındaki bir hedefe sızması.
- Güven modeli belgesi ihlalleri: bu politikanın, Hermes Agent'in
  kendi belgelerinin veya makul işletmeci beklentilerinin
  öngöreceğinin aksine davranan kod — Hermes Agent'in çıktısının
  tüketici bir katman (gösterge paneli, ağ geçidi adaptörü, dosya
  yazıcı, kabuk) tarafından nasıl işlenmesi gerektiğine dair bir
  duruş belgelemesi ve bir kod yolunun bu duruşu bozduğu durumlar
  dahil.

### 3.2 Kapsam Dışında

Buradaki "kapsam dışı", "bu politika kapsamında bir güvenlik açığı
değildir" anlamına gelir. "Bildirmeye değmez" anlamına gelmez. Süreç
içi sezgisel yöntemlerdeki iyileştirmeler, sertleştirme fikirleri ve
kullanıcı deneyimi (UX) düzeltmeleri, normal sorunlar (issue) veya
çekme talepleri (pull request) olarak memnuniyetle karşılanır — onay
kapısı her zaman daha fazla desen yakalayabilir, düzeltme her zaman
daha akıllı hale gelebilir, adaptör davranışı her zaman sıkılaştırılabilir.
Bu öğeler yalnızca özel ifşa kanalından geçmez ve güvenlik danışma bildirimi
(advisory) almaz.

- **Süreç içi sezgisel yöntemlerin atlatılması (Bölüm 2.4)** — onay
  kapısı regex atlatmaları, düzeltme atlatmaları, Yetenekler
  Koruyucusu desen atlatmaları ve gelecekteki sezgisel yöntemlere
  karşı benzer raporlar. Bu bileşenler sınır değildir; onları alt
  etmek bu politika kapsamında bir güvenlik açığı değildir.
- **Başlı başına komut enjeksiyonu (Prompt injection per se).** YDM'nin
  olağandışı çıktı üretmesini sağlamak — enjekte edilmiş içerik,
  halüsinasyon, eğitim yapay öğeleri (training artifacts) veya
  başka herhangi bir nedenle — kendi başına bir güvenlik açığı
  değildir. Zincirleme bir Bölüm 3.1 sonucu olmadan "komut
  enjeksiyonu gerçekleştirdim" ifadesi, bu politika kapsamında
  eyleme dönüştürülebilir bir rapor değildir.
- **Seçilen bir izolasyon duruşunun sonuçları.** Duruşunun kapsamı
  içinde çalışan bir kod yolunun, o duruşun izin verdiğini
  yapabileceğine dair raporlar güvenlik açığı değildir. Örnekler:
  yerel arka uç altında kabuk veya dosya araçlarının ana bilgisayar
  durumuna erişmesi; yalnızca kabuğu kum havuzuna alan terminal-arka
  ucu izolasyonu altında kod-yürütme veya MCP alt süreçlerinin ana
  bilgisayar durumuna erişmesi; ön koşulları, işletmeciye ait
  yapılandırma veya kimlik bilgisi dosyalarına önceden var olan yazma
  erişimi gerektiren raporlar (bunlar zaten güven zarfının içindedir).
- **Belgelenmiş kırılma-camı (break-glass) ayarları.** Korumaları
  açıkça devre dışı bırakan işletmeci tarafından seçilmiş ödünleşimler:
  `--insecure` ve gösterge paneli veya diğer bileşenlerdeki eşdeğer
  bayraklar, devre dışı bırakılmış onaylar, üretimde yerel arka uç,
  hermes-home güvenliğini atlayan geliştirme profilleri ve benzerleri.
  Bu yapılandırmalara karşı raporlar güvenlik açığı değildir — bu,
  bayrağın işidir.
- **Topluluk tarafından katkıda bulunulan yetenekler (skills) ve
  eklentiler (plugins).** Üçüncü taraf yetenekler (topluluk yetenek
  deposu dahil) ve üçüncü taraf eklentiler, Hermes Agent'in güven
  yüzeyinde değil (Bölüm 2.4, Bölüm 2.5), işletmecinin inceleme
  yüzeyindedir. Bir yetenek veya eklentinin kötü niyetli bir şey
  yapması, incelenmemiş bir yeteneğin beklenen başarısızlık şeklidir,
  Hermes Agent'de bir güvenlik açığı değildir. Hermes Agent'in
  yetenek-kurulum veya eklenti-kurulum yolundaki, işletmecinin ne
  yüklediğini görmesini engelleyen hatalar, Bölüm 3.1 kapsamındadır.
- **Harici kontroller olmadan genel sunum.** Ağ geçidini veya API'yi
  kimlik doğrulama, VPN veya güvenlik duvarı olmadan genel internete
  sunmak.
- **Kabuğa izin verilen bir duruşta araç düzeyinde okuma/yazma
  kısıtlamaları.** Bir yola terminal aracı aracılığıyla
  erişilebiliyorsa, diğer dosya araçlarının da bu yola
  erişebileceğine dair raporlar hiçbir şey eklemez.

---

## 4. Dağıtım Sertleştirme

En önemli tek sertleştirme kararı, izolasyonu (Bölüm 2.2) ajanın
alacağı içeriğin güvenilirliğiyle eşleştirmektir. Bunun ötesinde:

- Ajanı kök (root) olmayan bir kullanıcı olarak çalıştırın. Sağlanan
  konteyner imajı bunu varsayılan olarak yapar.
- Kimlik bilgilerini, sıkı izinlerle işletmeci kimlik bilgisi
  dosyasında tutun, asla ana yapılandırmada, asla sürüm kontrolünde
  tutmayın. OpenShell altında, diskteki bir kimlik bilgisi dosyası
  yerine Sağlayıcı deposunu (Provider store) kullanın.
- Ağ geçidini veya API'yi VPN, Tailscale veya güvenlik duvarı
  koruması olmadan genel internete sunmayın. OpenShell altında,
  giden trafiği (egress) kısıtlamak için ağ politikası katmanını
  kullanın.
- Etkinleştirdiğiniz her ağa açık adaptör için bir arayan izin listesi
  yapılandırın (Bölüm 2.6).
- Kurulumdan önce üçüncü taraf yetenekleri ve eklentileri inceleyin
  (Bölüm 2.4, Bölüm 2.5). Yetenekler için bu, yalnızca SKILL.md'yi
  değil, Python ve betikleri okumak anlamına gelir. Yetenekler
  Koruyucusu raporları ve kurulum denetim günlüğü, inceleme
  yüzeyidir.
- Hermes Agent, MCP sunucu başlatmaları ve CI'daki bağımlılık /
  paketlenmiş paket değişiklikleri için tedarik zinciri korumaları
  içerir; ayrıntılar için `CONTRIBUTING.md` dosyasına bakın.

---

## 5. İfşa

- **Koordineli ifşa penceresi (Coordinated disclosure window):** Rapordan
  itibaren 90 gün veya hangisi önce gelirse, bir düzeltme (fix)
  yayınlanana kadar.
- **Kanal:** GHSA başlığı veya security@nousresearch.com adresi ile
  e-posta yazışması.
- **Atıf (Credit):** Anonimlik talep edilmedikçe, bildirenler sürüm
  notlarında belirtilir.
