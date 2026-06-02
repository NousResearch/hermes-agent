package main

import "fmt"

func main() {
	fmt.Println("Running...")
}

func handleRequest(w http.ResponseWriter, r *http.Request) {
	fmt.Fprint(w, "OK")
}

func processData(input string) string {
	return input
}
