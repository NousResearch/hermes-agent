#!/usr/bin/env python3
"""
Questionário de Personalização do Hermes
Aprenda mais sobre suas preferências para personalizar serviços.
"""
import json
import os
from pathlib import Path

PREFERENCES_FILE = Path.home() / ".hermes" / "user_preferences.json"

def load_preferences():
    if PREFERENCES_FILE.exists():
        return json.loads(PREFERENCES_FILE.read_text())
    return {}

def save_preferences(prefs):
    PREFERENCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCES_FILE.write_text(json.dumps(prefs, indent=2, ensure_ascii=False))

def ask_question(question, options=None, multi=False):
    """Pergunta interativa simples."""
    print(f"\n{'='*60}")
    print(question)
    if options:
        for i, opt in enumerate(options, 1):
            print(f"  {i}. {opt}")
        if multi:
            print("  (Selecione vários números separados por vírgula, ou 0 para pular)")
        else:
            print("  (Digite o número ou 0 para pular)")
    
    while True:
        resp = input("\n→ ").strip()
        if resp == "0" or resp == "":
            return None
        if options:
            try:
                nums = [int(x.strip()) for x in resp.split(",") if x.strip()]
                if multi:
                    valid = [options[n-1] for n in nums if 1 <= n <= len(options)]
                    if valid:
                        return valid
                else:
                    n = int(resp)
                    if 1 <= n <= len(options):
                        return options[n-1]
            except ValueError:
                pass
        print("✗ Resposta inválida. Tente novamente.")

def main():
    print("\n" + "🌟" * 30)
    print("   QUESTIONÁRIO DE PERSONALIZAÇÃO DO HERMES")
    print("🌟" * 30)
    print("\nVou fazer algumas perguntas para personalizar melhor")
    print("suas experiências com o Hermes. Responda à vontade!")
    
    prefs = load_preferences()
    
    # Seção 1: Notícias de IA
    print("\n\n📬 === NOTÍCIAS E CONTEÚDO ===")
    prefs['news'] = prefs.get('news', {})
    
    topicos = ask_question(
        "📌 Quais TÓPICOS de notícias você quer receber? (selecione vários)",
        [
            "Inteligência Artificial / Machine Learning",
            "Ciência e descobertas recentes",
            "Tecnologia e gadgets",
            "Negócios e mercado",
            "Programação / Desenvolvimento",
            "Startup e inovação",
            "Saúde e biotech",
            "Espaço e astronomia",
            "Meio ambiente / sustentabilidade",
            "Política e sociedade",
            "Outros (especifique abaixo)"
        ],
        multi=True
    )
    if topicos:
        prefs['news']['topics'] = topicos
    
    frequencia = ask_question(
        "📅 Com que frequência quer receber as notícias?",
        [
            "Diário (todas as manhãs às 8h)",
            "Dias úteis apenas (seg-sex 8h)",
            "Semanal (todo domingo)",
            "A cada duas semanas",
            "Quando houver notícias importantes (push)",
        ]
    )
    if frequencia:
        prefs['news']['frequency'] = frequencia
    
    formato = ask_question(
        "📝 Prefere o resumo em que formato?",
        [
            "Bullets curtos (o que tem agora)",
            "Resumo mais detalhado (2-3 linhas por notícia)",
            "Links apenas com título",
            "Misturar: bullets + links importantes",
        ]
    )
    if formato:
        prefs['news']['format'] = formato
    
    idioma = ask_question(
        "🌐 Idioma preferido para as notícias:",
        [
            "Apenas português",
            "Preferência por português, mas aceito inglês para notícias importants",
            "Misto: português para contexto local, inglês para internacional",
            "Inglês (para acompanhar fonte original)",
            "Não importa, traduza automaticamente",
        ]
    )
    if idioma:
        prefs['news']['language'] = idioma
    
    # Seção 2: Telegram
    print("\n\n💬 === TELEGRAM ===")
    prefs['telegram'] = prefs.get('telegram', {})
    
    hora = ask_question(
        "⏰ Hora preferida para receber notícias no Telegram:",
        [
            "Manhã (6h-9h)",
            "Meio-dia (11h-13h)",
            "Tarde (14h-17h)",
            "Noite (19h-22h)",
            "Mantém as 8h como está",
        ]
    )
    if hora:
        prefs['telegram']['preferred_hour'] = hora
    
    som = ask_question(
        "🔔 Quer notificação sonora/no push no Telegram?",
        [
            "Sim, sempre que chegar uma notícia",
            "Apenas para notícias marcadas como 'importante'",
            "Não, prefiro ler quando quiser (sem push)",
        ]
    )
    if som is not None:
        prefs['telegram']['notifications'] = som
    
    # Seção 3: Email
    print("\n\n📧 === EMAIL ===")
    prefs['email'] = prefs.get('email', {})
    
    email_notifier = ask_question(
        "📧 Quer receber algum resumo por email também?",
        [
            "Sim, diariamente (mesmo conteúdo do Telegram)",
            "Semanal apenas (resumo maior)",
            "Apenas notícias marcadas como 'para ler depois'",
            "Não, prefiro apenas Telegram",
        ]
    )
    if email_notifier:
        prefs['email']['receive'] = email_notifier
    
    # Seção 4: Comunicação
    print("\n\n🤖 === COMUNICAÇÃO ===")
    prefs['communication'] = prefs.get('communication', {})
    
    tom = ask_question(
        "🗣️ Tom de voz preferido nas respostas do Hermes:",
        [
            "Direto e objetivo (só o essencial)",
            "Amigável e casual (mais humano)",
            "Formal e profissional",
            "Educado mas sem enrolação",
            "Meu jeito: varia conforme o contexto",
        ]
    )
    if tom:
        prefs['communication']['tone'] = tom
    
    detalhe = ask_question(
        "📊 Nível de detalhe nas respostas:",
        [
            "Só o necessário, nada de enrolação",
            "Resumo rápido + oferecer mais detalhes se quiser",
            "Detalhado por padrão, posso pedir para resumir",
            "Muito detalhado técnico",
        ]
    )
    if detalhe:
        prefs['communication']['detail_level'] = detalhe
    
    # Seção 5: Prioridades
    print("\n\n⚡ === PRIORIDADES ===")
    prefs['priorities'] = prefs.get('priorities', {})
    
    foco = ask_question(
        "🎯 Qual seu FOCO principal hoje? (o que é mais importante para você)",
        [
            "Trabalho / produtividade",
            "Aprendizado / estudos",
            "Projetos pessoais",
            "Saúde e bem-estar",
            "Família e relacionamentos",
            "Dinheiro / finanças",
            "Lazer e hobbies",
            "Various / não tenho foco específico",
        ]
    )
    if foco:
        prefs['priorities']['main_focus'] = foco
    
    metas = ask_question(
        "📈 Quais são suas PRIORIDADES para os próximos 3 meses? (selecione até 3)",
        [
            "Concluir um projeto específico",
            "Aprender algo novo",
            "Organizar minha rotina",
            "Melhorar minhas finanças",
            "Ficar mais saudável",
            "Conectar com pessoas",
            "Avançar na carreira",
            "Descansar mais / evitar burnout",
            "Não tenho metas definidas agora",
        ],
        multi=True
    )
    if metas:
        prefs['priorities']['goals'] = metas
    
    # Seção 6: Informações extras
    print("\n\n💡 === INFORMAÇÕES EXTRAS ===")
    
    trabalho = ask_question(
        "💼 Atuou você no trabalho? (opcional)",
        [
            "Sim, trabalho full-time",
            "Autônomo / freelancer",
            "Empreendedor / tenho minha própria empresa",
            "Estudante",
            "Investidor / aporta capital",
            "Não trabalho atualmente",
            "Prefiro não dizer",
        ]
    )
    if trabalho:
        prefs['work'] = {'status': trabalho}
    
    area = ask_question(
        "🔬 Qual sua área de atuação ou interesse principal? (opcional)",
        [
            "Tecnologia / TI / Programação",
            "Design / UX / Criativo",
            "Negócios / Gestão",
            "Ciências / Pesquisa",
            "Saúde / Medicina",
            "Educação",
            "Artes / Música",
            "Varejo / Comércio",
            "Legal / Jurídico",
            "Outro (especifique)",
        ]
    )
    if area:
        prefs['work']['area'] = area
    
    # Salvar
    save_preferences(prefs)
    
    print("\n\n" + "✅" * 30)
    print("   PERFEITO! Preferências salvas com sucesso!")
    print("✅" * 30)
    print("\nO Hermes agora tem mais contexto sobre você.")
    print("Nas próximas interações, vou usar essas preferências")
    print("para personalizar melhor suas experiências.")
    print("\n💡 Dica: Você pode corrigir ou atualizar qualquer")
    print("   preferência a qualquer momento apenas me pedindo.")

if __name__ == "__main__":
    main()
