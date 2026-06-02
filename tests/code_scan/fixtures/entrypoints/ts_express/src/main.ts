import express from 'express';

const app = express();
const PORT = 3000;

app.get('/', (_req, res) => {
    res.json({ message: 'Hello' });
});

app.listen(PORT, () => {
    console.log(`Listening on ${PORT}`);
});
