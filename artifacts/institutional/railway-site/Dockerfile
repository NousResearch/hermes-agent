FROM python:3.12-alpine
WORKDIR /app
COPY public ./public
EXPOSE 8080
CMD ["sh", "-c", "python -m http.server ${PORT:-8080} --bind 0.0.0.0 --directory /app/public"]
